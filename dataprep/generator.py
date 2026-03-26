import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("dataprep.generator")

MODEL_REGISTRY = {
    "tinyllama-1.1b": {
        "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "n_ctx": 2048,
    },
    "mistral-7b": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "n_ctx": 4096,
    },
}

SELF_INSTRUCT_PROMPT = """You are a training data generator. Given the following text, create exactly {n} instruction-response pairs for fine-tuning a language model on this content.

Rules:
- Each instruction must be a clear question or task about the text
- Each response must be accurate, detailed, and self-contained
- Cover different aspects: facts, explanations, summaries, analysis
- Output ONLY a valid JSON array, no other text

Text:
\"\"\"
{text}
\"\"\"

JSON array of {n} pairs:
[{{"instruction": "...", "output": "..."}}]"""

GENSTRUCT_PROMPT = """[INST] Given the following passage, generate a single high-quality question and comprehensive answer based on its content.

Passage:
\"\"\"
{text}
\"\"\"

Output exactly one JSON object:
{{"instruction": "your question", "output": "your answer"}} [/INST]"""


class DatasetGenerator:
    def __init__(self, model_dir: str, model_name: str = "mistral-7b",
                 n_threads: int = 4, n_ctx: int = 4096):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.n_threads = n_threads
        self.n_ctx = n_ctx
        self._llm = None
        self._loaded_model_name = None

    def _ensure_model(self) -> Path:
        info = MODEL_REGISTRY.get(self.model_name)
        if not info:
            raise ValueError(
                f"Unknown model: {self.model_name}. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        model_path = self.model_dir / info["filename"]
        if model_path.exists():
            logger.info(f"Model '{self.model_name}' found in cache")
            return model_path

        logger.info(f"Downloading model '{self.model_name}' from HuggingFace...")
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=str(self.model_dir),
        )
        logger.info(f"Model downloaded to {downloaded}")
        return Path(downloaded)

    def load_model(self):
        if self._loaded_model_name == self.model_name and self._llm is not None:
            return

        self.unload()
        model_path = self._ensure_model()
        info = MODEL_REGISTRY.get(self.model_name, {})

        logger.info(f"Loading model '{self.model_name}' from {model_path}")
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=info.get("n_ctx", self.n_ctx),
            n_threads=self.n_threads,
            verbose=False,
        )
        self._loaded_model_name = self.model_name
        logger.info("Model loaded")

    def generate_pairs(self, chunk: str, strategy: str = "self-instruct",
                       pairs_per_chunk: int = 3) -> list[dict]:
        self.load_model()

        if strategy == "genstruct":
            return self._generate_genstruct(chunk, pairs_per_chunk)
        return self._generate_self_instruct(chunk, pairs_per_chunk)

    def _generate_self_instruct(self, chunk: str, n: int) -> list[dict]:
        prompt = SELF_INSTRUCT_PROMPT.format(text=chunk[:3000], n=n)

        output = self._llm(prompt, max_tokens=1024, temperature=0.7, stop=["\n\n\n"])
        raw = output["choices"][0]["text"].strip()
        return self._parse_json_array(raw)

    def _generate_genstruct(self, chunk: str, n: int) -> list[dict]:
        pairs = []
        for _ in range(n):
            prompt = GENSTRUCT_PROMPT.format(text=chunk[:2000])

            output = self._llm(prompt, max_tokens=512, temperature=0.8, stop=["\n\n\n"])
            raw = output["choices"][0]["text"].strip()
            parsed = self._parse_single_pair(raw)
            if parsed:
                pairs.append(parsed)
        return pairs

    def _parse_json_array(self, raw: str) -> list[dict]:
        # Try full JSON array
        try:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                arr = json.loads(match.group())
                return [p for p in arr if "instruction" in p and "output" in p]
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: find individual JSON objects
        pairs = []
        for match in re.finditer(r'\{[^{}]*"instruction"[^{}]*"output"[^{}]*\}', raw, re.DOTALL):
            try:
                obj = json.loads(match.group())
                if "instruction" in obj and "output" in obj:
                    pairs.append(obj)
            except json.JSONDecodeError:
                continue

        if not pairs:
            logger.warning(f"Failed to parse pairs from: {raw[:200]}...")
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

        logger.warning(f"Failed to parse pair from: {raw[:200]}...")
        return None

    def unload(self):
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._loaded_model_name = None
            logger.info("Model unloaded")
