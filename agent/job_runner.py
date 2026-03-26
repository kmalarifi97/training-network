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
                result = await self._mock_finetune(job, progress_callback)
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

    async def _mock_finetune(self, job: dict, progress_callback=None) -> dict:
        """Mock fine-tuning — will be replaced with real LoRA training later."""
        job_id = job["job_id"]
        total_steps = 10

        for step in range(total_steps):
            if self._cancel_flag:
                return {"status": "cancelled", "reason": "Gamer returned", "last_step": step}
            if progress_callback:
                await progress_callback(job_id, f"Training step {step + 1}/{total_steps}")
            await asyncio.sleep(1)

        logger.info(f"Fine-tune job {job_id[:8]} completed")
        return {"status": "completed", "result": f"Fine-tuned model saved (mock, {total_steps} steps)"}

    def cancel(self):
        if self.current_job:
            logger.info(f"Cancelling job {self.current_job['job_id'][:8]} — gamer returned")
            self._cancel_flag = True
