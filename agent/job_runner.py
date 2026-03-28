import asyncio
import json
import logging
import re
from pathlib import Path

from llama_cpp import Llama

logger = logging.getLogger("agent.job_runner")

# Model registry: name -> (HuggingFace repo, filename, context size)
MODEL_REGISTRY = {
    "tinyllama-1.1b": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "n_ctx": 512,
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "n_ctx": 4096,
    },
    "llama3-8b": {
        "repo_id": "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
        "filename": "Meta-Llama-3-8B-Instruct.Q4_K_M.gguf",
        "n_ctx": 4096,
    },
}


class JobRunner:
    def __init__(self, model_cache_dir: str):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.current_job = None
        self._cancel_flag = False
        self._loaded_model_name = None
        self._llm = None

    @property
    def is_running(self) -> bool:
        return self.current_job is not None

    def _ensure_model(self, model_name: str) -> Path:
        """Return path to model file, downloading from HuggingFace if missing."""
        info = MODEL_REGISTRY.get(model_name)
        if not info:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

        model_path = self.model_cache_dir / info["filename"]
        if model_path.exists():
            logger.info(f"Model '{model_name}' found in cache: {model_path}")
            return model_path

        # Download from HuggingFace
        logger.info(f"Downloading model '{model_name}' from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id=info["repo_id"],
                filename=info["filename"],
                local_dir=str(self.model_cache_dir),
            )
            logger.info(f"Model downloaded to {downloaded}")
            return Path(downloaded)
        except ImportError:
            raise RuntimeError(
                "huggingface_hub not installed. Run: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model '{model_name}': {e}")

    def _load_model(self, model_name: str):
        # Reuse if same model is already loaded
        if self._loaded_model_name == model_name and self._llm is not None:
            logger.info(f"Model '{model_name}' already loaded, reusing")
            return

        # Unload previous model
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded_model_name = None

        # Ensure model is downloaded
        model_path = self._ensure_model(model_name)

        info = MODEL_REGISTRY.get(model_name, {})
        n_ctx = info.get("n_ctx", 512)

        logger.info(f"Loading model '{model_name}' from {model_path}")
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=4,
            verbose=False,
        )
        self._loaded_model_name = model_name
        logger.info(f"Model '{model_name}' loaded successfully")

    async def run_job(self, job: dict, progress_callback=None) -> dict:
        self.current_job = job
        self._cancel_flag = False
        job_id = job["job_id"]
        job_type = job.get("job_type", "inference")
        model_name = job.get("model_name", "tinyllama-1.1b")

        logger.info(f"Starting job {job_id[:8]}: {job_type} / {model_name}")

        try:
            if job_type == "inference":
                result = await self._run_inference(job, progress_callback)
            elif job_type == "dataprep":
                result = await self._run_dataprep(job, progress_callback)
            elif job_type == "fine-tune":
                result = await self._run_finetune(job, progress_callback)
            else:
                result = {"status": "failed", "error": f"Unknown job type: {job_type}"}
        except asyncio.CancelledError:
            result = {"status": "cancelled", "reason": "Task cancelled"}
        except Exception as e:
            logger.error(f"Job {job_id[:8]} failed: {e}")
            result = {"status": "failed", "error": str(e)}
        finally:
            self.current_job = None

        return result

    async def _run_inference(self, job: dict, progress_callback=None) -> dict:
        job_id = job["job_id"]
        model_name = job.get("model_name", "tinyllama-1.1b")
        prompt = job.get("prompt", "Hello")
        max_tokens = job.get("max_tokens", 128)

        # Download model if needed
        if progress_callback:
            await progress_callback(job_id, "Checking model cache...")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ensure_model, model_name)

        if self._cancel_flag:
            return {"status": "cancelled", "reason": "Gamer returned"}

        # Load model
        if progress_callback:
            await progress_callback(job_id, "Loading model...")

        await loop.run_in_executor(None, self._load_model, model_name)

        if self._cancel_flag:
            return {"status": "cancelled", "reason": "Gamer returned"}

        # Run inference in a thread (it's blocking)
        if progress_callback:
            await progress_callback(job_id, "Generating response...")

        def do_inference():
            output = self._llm(prompt, max_tokens=max_tokens, stop=["\n\n"])
            return output["choices"][0]["text"].strip()

        result_text = await loop.run_in_executor(None, do_inference)

        if self._cancel_flag:
            return {"status": "cancelled", "reason": "Gamer returned"}

        if progress_callback:
            await progress_callback(job_id, "Complete")

        logger.info(f"Job {job_id[:8]} completed: {result_text[:80]}...")
        return {"status": "completed", "result": result_text}

    # --- Dataprep: text → chunks → instruction pairs (all on GPU) ---

    SELF_INSTRUCT_PROMPT = (
        "You are a training data generator. Given the following text, create exactly {n} "
        "instruction-response pairs for fine-tuning a language model on this content.\n\n"
        "Rules:\n"
        "- Each instruction must be a clear question or task about the text\n"
        "- Each response must be accurate, detailed, and self-contained\n"
        "- Cover different aspects: facts, explanations, summaries, analysis\n"
        "- Output ONLY a valid JSON array, no other text\n\n"
        "Text:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "JSON array of {n} pairs:\n"
        '[{{"instruction": "...", "output": "..."}}]'
    )

    GENSTRUCT_PROMPT = (
        "[INST] Given the following passage, generate a single high-quality question "
        "and comprehensive answer based on its content.\n\n"
        "Passage:\n\"\"\"\n{text}\n\"\"\"\n\n"
        "Output exactly one JSON object:\n"
        '{{"instruction": "your question", "output": "your answer"}} [/INST]'
    )

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text).strip()
        if not text:
            return []
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_str = " ".join(words[start:end])
            if end < len(words):
                bp = max(chunk_str.rfind('. '), chunk_str.rfind('\n'))
                if bp > len(chunk_str) * 0.5:
                    chunk_str = chunk_str[:bp + 1].strip()
            if chunk_str:
                chunks.append(chunk_str)
            start = end - overlap
            if start >= len(words):
                break
        return chunks

    def _generate_pairs_from_chunk(self, chunk: str, strategy: str, n: int) -> list[dict]:
        if strategy == "genstruct":
            return self._genstruct_pairs(chunk, n)
        return self._self_instruct_pairs(chunk, n)

    def _self_instruct_pairs(self, chunk: str, n: int) -> list[dict]:
        prompt = self.SELF_INSTRUCT_PROMPT.format(text=chunk[:3000], n=n)
        output = self._llm(prompt, max_tokens=1024, temperature=0.7, stop=["\n\n\n"])
        raw = output["choices"][0]["text"].strip()
        return self._parse_json_pairs(raw)

    def _genstruct_pairs(self, chunk: str, n: int) -> list[dict]:
        pairs = []
        for _ in range(n):
            prompt = self.GENSTRUCT_PROMPT.format(text=chunk[:2000])
            output = self._llm(prompt, max_tokens=512, temperature=0.8, stop=["\n\n\n"])
            raw = output["choices"][0]["text"].strip()
            parsed = self._parse_single_pair(raw)
            if parsed:
                pairs.append(parsed)
        return pairs

    def _parse_json_pairs(self, raw: str) -> list[dict]:
        try:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                arr = json.loads(match.group())
                return [p for p in arr if "instruction" in p and "output" in p]
        except (json.JSONDecodeError, TypeError):
            pass
        pairs = []
        for match in re.finditer(r'\{[^{}]*"instruction"[^{}]*"output"[^{}]*\}', raw, re.DOTALL):
            try:
                obj = json.loads(match.group())
                if "instruction" in obj and "output" in obj:
                    pairs.append(obj)
            except json.JSONDecodeError:
                continue
        return pairs

    def _parse_single_pair(self, raw: str) -> dict | None:
        try:
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                obj = json.loads(match.group())
                if "instruction" in obj and "output" in obj:
                    return obj
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    async def _run_dataprep(self, job: dict, progress_callback=None) -> dict:
        """Full dataset generation on GPU: chunk text, generate pairs, return all."""
        job_id = job["job_id"]
        model_name = job.get("model_name", "mistral-7b")

        # Parse the job prompt which contains all dataprep parameters
        try:
            params = json.loads(job.get("prompt", "{}"))
        except json.JSONDecodeError:
            return {"status": "failed", "error": "Invalid dataprep job parameters"}

        text = params.get("text", "")
        strategy = params.get("strategy", "self-instruct")
        pairs_per_chunk = params.get("pairs_per_chunk", 3)

        if not text.strip():
            return {"status": "failed", "error": "No text provided"}

        # Step 1: Chunk
        if progress_callback:
            await progress_callback(job_id, "Chunking text...")
        chunks = self._chunk_text(text)
        if not chunks:
            return {"status": "failed", "error": "Text too short to chunk"}

        logger.info(f"Job {job_id[:8]}: {len(chunks)} chunks from {len(text)} chars")

        # Step 2: Load model
        if progress_callback:
            await progress_callback(job_id, "Loading model...")

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._ensure_model, model_name)
        if self._cancel_flag:
            return {"status": "cancelled", "reason": "Gamer returned"}

        await loop.run_in_executor(None, self._load_model, model_name)
        if self._cancel_flag:
            return {"status": "cancelled", "reason": "Gamer returned"}

        # Step 3: Generate pairs from each chunk on GPU
        all_pairs = []
        for i, chunk in enumerate(chunks):
            if self._cancel_flag:
                return {"status": "cancelled", "reason": "Gamer returned",
                        "partial_pairs": all_pairs}

            if progress_callback:
                await progress_callback(job_id,
                    f"Generating pairs: chunk {i+1}/{len(chunks)}")
                # Send structured progress for dataset tracking
                if hasattr(progress_callback, '__self__'):
                    pass  # handled via dataprep_progress below

            def gen_chunk(c=chunk):
                return self._generate_pairs_from_chunk(c, strategy, pairs_per_chunk)

            pairs = await loop.run_in_executor(None, gen_chunk)
            all_pairs.extend(pairs)
            logger.info(f"Job {job_id[:8]}: chunk {i+1}/{len(chunks)} → {len(pairs)} pairs")

        if not all_pairs:
            return {"status": "failed", "error": "Model produced no usable pairs"}

        if progress_callback:
            await progress_callback(job_id, "Complete")

        logger.info(f"Job {job_id[:8]}: dataprep done — {len(all_pairs)} total pairs")
        result = json.dumps({"pairs": all_pairs, "total_pairs": len(all_pairs)})
        return {"status": "completed", "result": result}

    # --- Fine-tuning: real QLoRA training via peft + transformers ---

    FINETUNE_ALPACA_TEMPLATE = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n{output}"
    )

    async def _run_finetune(self, job: dict, progress_callback=None) -> dict:
        """Real QLoRA fine-tuning using peft + transformers + trl."""
        job_id = job["job_id"]

        try:
            params = json.loads(job.get("prompt", "{}"))
        except json.JSONDecodeError:
            return {"status": "failed", "error": "Invalid fine-tune job parameters"}

        run_id = params.get("run_id", "unknown")
        hf_model_id = params.get("hf_model_id", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        dataset_url = params.get("dataset_url", "")
        adapter_upload_url = params.get("adapter_upload_url", "")
        epochs = params.get("epochs", 3)
        batch_size = params.get("batch_size", 4)
        learning_rate = params.get("learning_rate", 2e-4)
        lora_r = params.get("lora_r", 16)
        lora_alpha = params.get("lora_alpha", 32)
        max_steps = params.get("max_steps", -1)
        sample_count = params.get("sample_count", 0)

        logger.info(f"Fine-tune {run_id[:8]}: {hf_model_id}, {sample_count} samples, {epochs} epochs")

        # Step 1: Download training data from server
        if progress_callback:
            await progress_callback(job_id, "Downloading training data...")

        dataset_path = self.model_cache_dir / f"finetune_{run_id}.jsonl"
        try:
            await self._download_dataset(dataset_url, dataset_path)
        except Exception as e:
            return {"status": "failed", "error": f"Failed to download dataset: {e}"}

        if self._cancel_flag:
            return {"status": "cancelled", "reason": "Gamer returned"}

        # Step 2: Run training in a thread (blocking GPU work)
        if progress_callback:
            await progress_callback(job_id, "Loading model and starting training...")

        loop = asyncio.get_event_loop()
        training_log = []

        # Progress bridge: called from training thread, posts to async callback
        def on_train_step(step, total, loss, epoch):
            training_log.append({"step": step, "loss": loss, "epoch": epoch})
            if progress_callback:
                asyncio.run_coroutine_threadsafe(
                    progress_callback(
                        job_id,
                        f"Training step {step}/{total} | loss: {loss:.4f}",
                        finetune_progress={
                            "step": step,
                            "total_steps": total,
                            "loss": loss,
                            "epoch": epoch,
                            "log": training_log[-20:],  # last 20 entries
                        },
                    ),
                    loop,
                )

        adapter_output_dir = self.model_cache_dir / f"adapter_{run_id}"

        def do_training():
            return self._train_lora(
                hf_model_id=hf_model_id,
                dataset_path=str(dataset_path),
                output_dir=str(adapter_output_dir),
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                max_steps=max_steps,
                on_step=on_train_step,
            )

        try:
            result = await loop.run_in_executor(None, do_training)
        except Exception as e:
            logger.error(f"Fine-tune {run_id[:8]} failed: {e}")
            return {"status": "failed", "error": str(e)}

        if self._cancel_flag:
            return {"status": "cancelled", "reason": "Gamer returned"}

        # Step 3: Zip and upload adapter to server
        if progress_callback:
            await progress_callback(job_id, "Uploading adapter to server...")

        zip_path = self.model_cache_dir / f"adapter_{run_id}.zip"
        try:
            self._zip_adapter(str(adapter_output_dir), str(zip_path))
            await self._upload_adapter(adapter_upload_url, str(zip_path))
        except Exception as e:
            logger.warning(f"Adapter upload failed: {e} (adapter saved locally at {adapter_output_dir})")

        if progress_callback:
            await progress_callback(job_id, "Complete")

        final_loss = training_log[-1]["loss"] if training_log else None
        total_steps = training_log[-1]["step"] if training_log else 0

        logger.info(f"Fine-tune {run_id[:8]} completed: {total_steps} steps, final loss={final_loss}")

        result_json = json.dumps({
            "adapter_path": str(adapter_output_dir),
            "final_loss": final_loss,
            "total_steps": total_steps,
            "log": training_log,
        })
        return {"status": "completed", "result": result_json}

    def _train_lora(self, hf_model_id: str, dataset_path: str, output_dir: str,
                    epochs: int, batch_size: int, learning_rate: float,
                    lora_r: int, lora_alpha: int, max_steps: int,
                    on_step=None) -> dict:
        """Blocking function: loads model, trains LoRA, saves adapter. Runs in thread."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTTrainer, SFTConfig
        from datasets import load_dataset
        import transformers

        logger.info(f"Loading tokenizer: {hf_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model in 4-bit for QLoRA
        logger.info(f"Loading model in 4-bit: {hf_model_id}")
        use_4bit = torch.cuda.is_available()
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model)
        else:
            # CPU fallback — float32, no quantization
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        # Configure LoRA adapter
        logger.info(f"Configuring LoRA: r={lora_r}, alpha={lora_alpha}")
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        # Load dataset
        logger.info(f"Loading dataset: {dataset_path}")
        dataset = load_dataset("json", data_files=dataset_path, split="train")

        # Format into text column
        def format_sample(example):
            text = self.FINETUNE_ALPACA_TEMPLATE.format(
                instruction=example.get("instruction", ""),
                input=example.get("input", ""),
                output=example.get("output", ""),
            )
            return {"text": text}

        dataset = dataset.map(format_sample)

        # Custom callback for progress
        class ProgressCallback(transformers.TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if on_step and state.global_step > 0:
                    loss = logs.get("loss", 0) if logs else 0
                    on_step(state.global_step, state.max_steps, loss, state.epoch or 0)

        # Training arguments
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=max(1, 4 // batch_size),
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=1,
            save_strategy="epoch",
            max_steps=max_steps if max_steps > 0 else -1,
            fp16=torch.cuda.is_available(),
            optim="adamw_torch",
            max_seq_length=512,
            dataset_text_field="text",
            report_to="none",
        )

        # Train
        logger.info("Starting training...")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[ProgressCallback()],
        )

        trainer.train()

        # Save adapter only (not the full model)
        logger.info(f"Saving adapter to {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Clean up GPU memory
        del model
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"output_dir": output_dir}

    async def _download_dataset(self, url_path: str, dest: Path):
        """Download training dataset from server via HTTP."""
        import httpx
        # Build full URL from the agent's server connection
        server_ws = self.model_cache_dir  # We'll get the base URL from the job
        # The URL path is relative — build from the server's HTTP base
        # Agent knows the WS URL, derive HTTP from it
        base_http = self._get_server_http_url()
        full_url = f"{base_http}{url_path}"
        logger.info(f"Downloading dataset from {full_url}")
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(full_url)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                f.write(resp.content)
        logger.info(f"Dataset saved to {dest} ({len(resp.content)} bytes)")

    def _get_server_http_url(self) -> str:
        """Derive HTTP base URL from the WebSocket URL in config."""
        from config import load_config
        config = load_config()
        ws_url = config["server_url"]
        # ws://host:port/agents/connect -> http://host:port
        # wss://host:port/agents/connect -> https://host:port
        http_url = ws_url.replace("wss://", "https://").replace("ws://", "http://")
        # Strip the path
        from urllib.parse import urlparse
        parsed = urlparse(http_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _zip_adapter(self, adapter_dir: str, zip_path: str):
        """Zip the adapter directory for upload."""
        import shutil
        shutil.make_archive(zip_path.replace(".zip", ""), "zip", adapter_dir)
        logger.info(f"Adapter zipped: {zip_path}")

    async def _upload_adapter(self, url_path: str, zip_path: str):
        """Upload adapter zip to server."""
        import httpx
        base_http = self._get_server_http_url()
        full_url = f"{base_http}{url_path}"
        logger.info(f"Uploading adapter to {full_url}")
        async with httpx.AsyncClient(timeout=300) as client:
            with open(zip_path, "rb") as f:
                resp = await client.post(
                    full_url,
                    files={"file": ("adapter.zip", f, "application/zip")},
                )
                resp.raise_for_status()
        logger.info("Adapter uploaded successfully")

    def cancel(self):
        if self.current_job:
            logger.info(f"Cancelling job {self.current_job['job_id'][:8]} — gamer returned")
            self._cancel_flag = True
