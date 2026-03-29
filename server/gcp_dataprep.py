"""
GCP-based dataset generation pipeline.
Document AI for PDF text extraction, Gemini for training pair generation.
Supports: PDF (Document AI + Gemini), CSV (Gemini), plain text (Gemini).
"""

import asyncio
import base64
import csv
import io
import json
import logging
import os
import re
import tempfile
from pathlib import Path

import httpx

logger = logging.getLogger("gcp_dataprep")

# --- Config ---
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "training-network-sa")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us")
DOCAI_PROCESSOR_ID = os.environ.get("DOCAI_PROCESSOR_ID", "242349cfad9c29c")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    print("[gcp_dataprep] WARNING: GEMINI_API_KEY is not set!")
else:
    print(f"[gcp_dataprep] Gemini API key loaded: {GEMINI_API_KEY[:10]}...")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

DOCAI_ENDPOINT = (
    f"https://{GCP_LOCATION}-documentai.googleapis.com/v1/projects/"
    f"{GCP_PROJECT_ID}/locations/{GCP_LOCATION}/processors/{DOCAI_PROCESSOR_ID}:process"
)
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

# Document AI page limit per request
DOCAI_PAGE_LIMIT = 15


# =============================================================================
# Step 1: Extract text via Document AI
# =============================================================================

async def extract_text_docai(pdf_bytes: bytes, access_token: str) -> str:
    """Send PDF to Document AI OCR, return cleaned text.
    Splits into batches of DOCAI_PAGE_LIMIT pages automatically."""

    # Count pages
    from PyPDF2 import PdfReader, PdfWriter
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    logger.info(f"PDF has {total_pages} pages")

    if total_pages == 0:
        return ""

    # Split into batches
    batches = []
    for start in range(0, total_pages, DOCAI_PAGE_LIMIT):
        end = min(start + DOCAI_PAGE_LIMIT, total_pages)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        buf = io.BytesIO()
        writer.write(buf)
        batches.append((start, end, buf.getvalue()))

    logger.info(f"Split into {len(batches)} batches for Document AI")

    # Process all batches concurrently
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = [
            _process_docai_batch(client, batch_bytes, access_token)
            for _, _, batch_bytes in batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge text
    all_text = []
    for i, result in enumerate(results):
        start, end, _ = batches[i]
        if isinstance(result, Exception):
            logger.error(f"Batch pages {start+1}-{end} failed: {result}")
            continue
        all_text.append(result)

    full_text = "\n".join(all_text)
    return _clean_text(full_text)


async def _process_docai_batch(client: httpx.AsyncClient, pdf_bytes: bytes,
                                access_token: str) -> str:
    """Send a single PDF batch to Document AI, return extracted text."""
    b64 = base64.b64encode(pdf_bytes).decode()
    payload = {
        "rawDocument": {"content": b64, "mimeType": "application/pdf"}
    }

    resp = await client.post(
        DOCAI_ENDPOINT,
        json=payload,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    resp.raise_for_status()
    data = resp.json()

    doc = data.get("document", {})
    return doc.get("text", "")


def _clean_text(text: str) -> str:
    """Remove common watermarks and page numbers."""
    text = re.sub(r'Twitter: @ketab_n\n?', '', text)
    text = re.sub(r'ketab\.me\n?', '', text)
    text = re.sub(r'\n[٠-٩۰-۹]{1,2}\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# =============================================================================
# Step 2: Chunk text
# =============================================================================

def chunk_text(text: str, max_chars: int = 2500) -> list[str]:
    """Split text into chunks of ~max_chars, breaking at paragraph boundaries."""
    paragraphs = text.split('\n')
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) > max_chars and len(current) > 500:
            chunks.append(current.strip())
            current = para
        else:
            current += "\n" + para

    if current.strip():
        chunks.append(current.strip())

    # Filter tiny chunks
    chunks = [c for c in chunks if len(c) > 200]
    logger.info(f"Split {len(text)} chars into {len(chunks)} chunks")
    return chunks


# =============================================================================
# Step 3: Generate training pairs via Gemini
# =============================================================================

GENERATION_PROMPT = """أنت خبير في إنشاء بيانات تدريب للنماذج اللغوية. من النص التالي، قم بإنشاء {n} أزواج تدريبية بصيغة JSONL.

كل زوج يجب أن يكون بنفس لغة النص ويتضمن:
- instruction: سؤال أو طلب واضح ومتنوع
- input: سياق من النص إذا لزم (أو نص فارغ "")
- output: إجابة دقيقة ومفصلة مبنية على النص

نوّع في أنواع الأسئلة: فهم، تلخيص، شرح مفاهيم، مقارنة، تحليل، استنتاج، تعريف مصطلحات.

النص:
{text}

أعد النتيجة كـ JSONL فقط (سطر JSON واحد لكل زوج). بدون أي شرح إضافي أو markdown أو code fences.
مهم: جميع القيم يجب أن تكون نصوص (strings) وليس كائنات أو مصفوفات.
كل سطر يجب أن يكون بالضبط: {{"instruction": "نص", "input": "نص", "output": "نص"}}"""


async def generate_pairs_gemini(chunks: list[str], pairs_per_chunk: int = 10,
                                 output_format: str = "alpaca",
                                 progress_callback=None) -> list[dict]:
    """Send each chunk to Gemini, collect instruction/response pairs."""
    all_pairs = []

    async with httpx.AsyncClient(timeout=120) as client:
        for i, chunk in enumerate(chunks):
            prompt = GENERATION_PROMPT.format(text=chunk, n=pairs_per_chunk)
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 8192,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }

            try:
                url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    print(f"[gemini] Chunk {i+1}: API error: {data['error'].get('message','?')}")
                    continue

                text = data["candidates"][0]["content"]["parts"][0]["text"]
                pairs = _parse_jsonl(text, output_format)
                all_pairs.extend(pairs)

                print(f"[gemini] Chunk {i+1}/{len(chunks)}: {len(pairs)} pairs (raw: {len(text)} chars)")

                if progress_callback:
                    await progress_callback(
                        processed_chunks=i + 1,
                        total_chunks=len(chunks),
                        total_pairs=len(all_pairs),
                    )

            except Exception as e:
                print(f"[gemini] Chunk {i+1}/{len(chunks)} EXCEPTION: {type(e).__name__}: {e}")

            # Small delay to avoid rate limits
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)

    return all_pairs


def _parse_jsonl(raw: str, output_format: str) -> list[dict]:
    """Parse Gemini output into training pairs.
    Handles: markdown fences, input as dict/list, thinking tags, etc."""
    # Strip markdown code fences
    text = re.sub(r'```(?:json|jsonl)?\s*\n?', '', raw)
    text = re.sub(r'```\s*$', '', text, flags=re.MULTILINE)

    # Strip <think>...</think> blocks (Gemini thinking)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    pairs = []

    # Strategy 1: line-by-line JSONL parsing
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            pair = _normalize_pair(obj, output_format)
            if pair:
                pairs.append(pair)
        except json.JSONDecodeError:
            continue

    # Strategy 2: if line-by-line failed, try finding JSON objects with regex
    if not pairs:
        for match in re.finditer(r'\{[^{}]*"instruction"[^{}]*\}', text, re.DOTALL):
            try:
                obj = json.loads(match.group())
                pair = _normalize_pair(obj, output_format)
                if pair:
                    pairs.append(pair)
            except json.JSONDecodeError:
                continue

    # Strategy 3: try parsing as a JSON array
    if not pairs:
        try:
            arr_match = re.search(r'\[.*\]', text, re.DOTALL)
            if arr_match:
                arr = json.loads(arr_match.group())
                for obj in arr:
                    pair = _normalize_pair(obj, output_format)
                    if pair:
                        pairs.append(pair)
        except (json.JSONDecodeError, TypeError):
            pass

    return pairs


def _normalize_pair(obj: dict, output_format: str) -> dict | None:
    """Normalize a single training pair, converting non-string values."""
    if not isinstance(obj, dict):
        return None
    if "instruction" not in obj or "output" not in obj:
        return None

    def to_str(val):
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        if not isinstance(val, str):
            return str(val)
        return val

    instruction = to_str(obj["instruction"])
    raw_input = to_str(obj.get("input", ""))
    raw_output = to_str(obj["output"])

    if output_format == "chat":
        return {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": raw_output},
            ]
        }
    return {
        "instruction": instruction,
        "input": raw_input,
        "output": raw_output,
    }


# =============================================================================
# Full pipeline
# =============================================================================

async def get_access_token() -> str:
    """Get GCP access token via gcloud CLI."""
    proc = await asyncio.create_subprocess_exec(
        "gcloud", "auth", "print-access-token",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to get access token: {stderr.decode()}")
    return stdout.decode().strip()


async def run_pipeline(pdf_bytes: bytes, pairs_per_chunk: int = 10,
                       output_format: str = "alpaca",
                       progress_callback=None) -> list[dict]:
    """Full pipeline: Document AI extraction -> chunk -> Gemini generation."""

    # Step 1: Extract text
    if progress_callback:
        await progress_callback(status="EXTRACTING", processed_chunks=0,
                                total_chunks=0, total_pairs=0)

    access_token = await get_access_token()
    text = await extract_text_docai(pdf_bytes, access_token)

    if not text.strip():
        raise ValueError("No text could be extracted from the PDF")

    logger.info(f"Extracted {len(text)} chars")

    # Step 2: Chunk
    chunks = chunk_text(text)

    if progress_callback:
        await progress_callback(status="GENERATING", processed_chunks=0,
                                total_chunks=len(chunks), total_pairs=0)

    # Step 3: Generate pairs
    pairs = await generate_pairs_gemini(
        chunks, pairs_per_chunk, output_format, progress_callback
    )

    return pairs


async def run_text_pipeline(text: str, pairs_per_chunk: int = 10,
                            output_format: str = "alpaca",
                            progress_callback=None) -> list[dict]:
    """Pipeline for raw text input (skip Document AI)."""
    chunks = chunk_text(text)

    if progress_callback:
        await progress_callback(status="GENERATING", processed_chunks=0,
                                total_chunks=len(chunks), total_pairs=0)

    pairs = await generate_pairs_gemini(
        chunks, pairs_per_chunk, output_format, progress_callback
    )

    return pairs


# =============================================================================
# CSV pipeline
# =============================================================================

CSV_ROWS_PER_BATCH = 20  # rows per Gemini call

CSV_GENERATION_PROMPT = """You are an expert at creating fine-tuning training data from structured/tabular data.

Below is a CSV dataset with these columns:
{columns}

Here are {row_count} sample rows:
{rows}

Generate exactly {n} high-quality instruction/response training pairs based on this data.

Each pair should be diverse. Include these types:
- Record analysis: "Analyze this record for anomalies/patterns"
- Classification: "Classify this record" or "What category does this belong to?"
- Comparison: "Compare these two records"
- Summarization: "Summarize the key patterns in these records"
- Reasoning: "Why might this record be flagged as X?"
- Data interpretation: "What does this data tell us about Y?"

Rules:
- Use actual values from the rows in your instructions and outputs
- Outputs must be detailed and analytical, not just restating the data
- Match the language of the data (if data is in Arabic, pairs should be in Arabic)
- Output ONLY valid JSONL (one JSON object per line)
- No markdown, no code fences, no explanation before or after
- CRITICAL: All values must be plain strings, NOT objects or arrays

Each line must be exactly: {{"instruction": "a string", "input": "a string", "output": "a string"}}"""


def parse_csv(csv_bytes: bytes) -> tuple[list[str], list[dict]]:
    """Parse CSV bytes into (columns, rows). Handles common encodings."""
    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1256"):
        try:
            text = csv_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = csv_bytes.decode("utf-8", errors="replace")

    reader = csv.DictReader(io.StringIO(text))
    columns = reader.fieldnames or []
    rows = list(reader)
    return columns, rows


def batch_csv_rows(columns: list[str], rows: list[dict],
                   batch_size: int = CSV_ROWS_PER_BATCH) -> list[str]:
    """Convert CSV rows into text batches for Gemini."""
    batches = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        # Format rows as readable text
        lines = []
        for row in batch_rows:
            line = " | ".join(f"{col}: {row.get(col, '')}" for col in columns)
            lines.append(line)
        batches.append("\n".join(lines))
    return batches


async def generate_pairs_csv_gemini(columns: list[str], row_batches: list[str],
                                     pairs_per_batch: int = 10,
                                     output_format: str = "alpaca",
                                     progress_callback=None) -> list[dict]:
    """Send CSV row batches to Gemini, collect training pairs."""
    all_pairs = []
    columns_str = ", ".join(columns)

    async with httpx.AsyncClient(timeout=120) as client:
        for i, batch in enumerate(row_batches):
            row_count = batch.count("\n") + 1
            prompt = CSV_GENERATION_PROMPT.format(
                columns=columns_str,
                row_count=row_count,
                rows=batch,
                n=pairs_per_batch,
            )
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 8192,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }

            try:
                url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    print(f"[gemini-csv] Batch {i+1}: API error: {data['error'].get('message','?')}")
                    continue

                text = data["candidates"][0]["content"]["parts"][0]["text"]
                pairs = _parse_jsonl(text, output_format)
                all_pairs.extend(pairs)

                print(f"[gemini-csv] Batch {i+1}/{len(row_batches)}: {len(pairs)} pairs (raw: {len(text)} chars)")
                if len(pairs) == 0:
                    print(f"[gemini-csv] RAW UNPARSED: {repr(text[:500])}")

                if progress_callback:
                    await progress_callback(
                        processed_chunks=i + 1,
                        total_chunks=len(row_batches),
                        total_pairs=len(all_pairs),
                    )

            except Exception as e:
                print(f"[gemini-csv] Batch {i+1}/{len(row_batches)} EXCEPTION: {type(e).__name__}: {e}")

            if i < len(row_batches) - 1:
                await asyncio.sleep(0.5)

    return all_pairs


async def run_csv_pipeline(csv_bytes: bytes, pairs_per_batch: int = 10,
                           output_format: str = "alpaca",
                           progress_callback=None) -> list[dict]:
    """Full CSV pipeline: parse -> batch rows -> Gemini generation."""
    columns, rows = parse_csv(csv_bytes)

    if not columns or not rows:
        raise ValueError("CSV is empty or could not be parsed")

    logger.info(f"CSV: {len(columns)} columns, {len(rows)} rows")

    row_batches = batch_csv_rows(columns, rows)

    if progress_callback:
        await progress_callback(status="GENERATING", processed_chunks=0,
                                total_chunks=len(row_batches), total_pairs=0)

    pairs = await generate_pairs_csv_gemini(
        columns, row_batches, pairs_per_batch, output_format, progress_callback
    )

    return pairs
