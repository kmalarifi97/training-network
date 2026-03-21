import asyncio
import logging
from pathlib import Path

from llama_cpp import Llama

logger = logging.getLogger("agent.job_runner")

# Map model names to local GGUF files
MODEL_FILES = {
    "tinyllama-1.1b": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
}


class JobRunner:
    def __init__(self, model_cache_dir: str):
        self.model_cache_dir = Path(model_cache_dir)
        self.current_job = None
        self._cancel_flag = False
        self._loaded_model_name = None
        self._llm = None

    @property
    def is_running(self) -> bool:
        return self.current_job is not None

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

        filename = MODEL_FILES.get(model_name)
        if not filename:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_FILES.keys())}")

        model_path = self.model_cache_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model '{model_name}' from {model_path}")
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=512,
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

        # Load model
        if progress_callback:
            await progress_callback(job_id, "Loading model...")

        # Run model loading in a thread (it's blocking)
        loop = asyncio.get_event_loop()
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
