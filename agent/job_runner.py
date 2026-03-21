import asyncio
import logging
import time

logger = logging.getLogger("agent.job_runner")


class JobRunner:
    def __init__(self):
        self.current_job = None
        self._cancel_flag = False

    @property
    def is_running(self) -> bool:
        return self.current_job is not None

    async def run_job(self, job: dict, progress_callback=None) -> dict:
        """
        Run a job. For now this is a MOCK that simulates work.
        Later this will be replaced with real inference/fine-tuning.

        Returns: {"status": "completed"/"cancelled"/"failed", "result": "...", ...}
        """
        self.current_job = job
        self._cancel_flag = False
        job_id = job["job_id"]
        job_type = job.get("job_type", "inference")
        model_name = job.get("model_name", "mock-model")
        prompt = job.get("prompt", "")

        logger.info(f"Starting job {job_id[:8]}: {job_type} / {model_name}")

        try:
            if job_type == "inference":
                result = await self._mock_inference(job, progress_callback)
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

    async def _mock_inference(self, job: dict, progress_callback=None) -> dict:
        """
        Mock inference — simulates loading a model and generating text.
        Will be replaced with real llama.cpp / vLLM calls later.
        """
        job_id = job["job_id"]
        prompt = job.get("prompt", "Hello")

        # Simulate model loading (2 seconds)
        if progress_callback:
            await progress_callback(job_id, "Loading model into GPU...")
        await asyncio.sleep(2)

        if self._cancel_flag:
            return {"status": "cancelled", "reason": "Gamer returned"}

        # Simulate token generation (5 seconds)
        if progress_callback:
            await progress_callback(job_id, "Generating response...")

        generated = f"[Mock response to: '{prompt}'] "
        for i in range(5):
            if self._cancel_flag:
                return {"status": "cancelled", "reason": "Gamer returned"}
            generated += f"Word{i + 1} "
            await asyncio.sleep(1)

        if progress_callback:
            await progress_callback(job_id, "Complete")

        logger.info(f"Job {job_id[:8]} completed")
        return {"status": "completed", "result": generated.strip()}

    async def _mock_finetune(self, job: dict, progress_callback=None) -> dict:
        """
        Mock fine-tuning — simulates training steps.
        """
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
