import asyncio
import logging
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
