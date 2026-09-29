"""Build resumable embeddings for the approved curriculum Markdown."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import zipfile

import numpy as np

from curriculum_rag import (DATASET_NAME, EMBEDDINGS_NAME, METADATA_NAME,
                            canonical_bytes, embedding_metadata, embedding_text,
                            parse_curriculum)
from rag import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, validate_vectors


APP_DIR = Path(__file__).resolve().parent
ITEMS_PER_WINDOW = 80
WINDOW_DELAY_SECONDS = 80
MAX_RETRIES = 3
CHECKPOINT_BINARY = ".curriculum_embedding_checkpoint.npz"
CHECKPOINT_METADATA = ".curriculum_embedding_checkpoint.json"


def replace_with_retry(source, destination):
    """Retry transient Windows locks on newly saved npz files."""
    for attempt in range(6):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (attempt + 1))


def retry_delay(error, retries):
    code = getattr(error, "code", None)
    if code not in (429, 503) and type(error).__name__ not in ("ResourceExhausted", "ServerError"):
        return None
    message = str(error)
    match = re.search(r"Please retry in\s+([0-9.]+)s", message, re.IGNORECASE)
    requested = math.ceil(float(match.group(1))) + 1 if match else 0
    return max(WINDOW_DELAY_SECONDS if code == 429 else min(20 * 2 ** retries, 80), requested)


def load_checkpoint(directory, source_bytes, count):
    try:
        metadata = json.loads((directory / CHECKPOINT_METADATA).read_text(encoding="utf-8"))
        expected = embedding_metadata(source_bytes, count)
        if any(metadata.get(key) != value for key, value in expected.items()):
            return None
        binary = directory / CHECKPOINT_BINARY
        if metadata.get("embeddings_sha256") != hashlib.sha256(binary.read_bytes()).hexdigest():
            return None
        with np.load(binary, allow_pickle=False) as data:
            vectors = data["embeddings"]
        completed = metadata.get("completed_count")
        if not isinstance(completed, int) or not 0 < completed <= count:
            return None
        return validate_vectors(vectors, completed)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, EOFError,
            zipfile.BadZipFile):
        return None


def save_checkpoint(directory, source_bytes, count, vectors):
    metadata = embedding_metadata(source_bytes, count)
    metadata["completed_count"] = len(vectors)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        binary = Path(temporary) / CHECKPOINT_BINARY
        meta = Path(temporary) / CHECKPOINT_METADATA
        np.savez_compressed(binary, embeddings=vectors)
        metadata["embeddings_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        replace_with_retry(binary, directory / binary.name)
        replace_with_retry(meta, directory / meta.name)


def build_embeddings(directory, embed_content, sleep_fn=time.sleep, progress=print,
                     pause_at_half=True):
    directory = Path(directory)
    source_bytes = canonical_bytes((directory / DATASET_NAME).read_bytes())
    records = parse_curriculum(source_bytes)
    texts = [embedding_text(record) for record in records]
    checkpoint = load_checkpoint(directory, source_bytes, len(texts))
    batches = [checkpoint] if checkpoint is not None else []
    start = len(checkpoint) if checkpoint is not None else 0
    if start:
        progress(f"Resuming curriculum embeddings at {start}/{len(texts)} sections")
    items_in_window = 0
    half = math.ceil(len(texts) / 2)
    halfway = half if pause_at_half and start <= half else None

    while start < len(texts):
        if halfway is not None and start == halfway:
            progress(f"Embedded first half ({start}/{len(texts)}); "
                     f"waiting {WINDOW_DELAY_SECONDS} seconds...")
            sleep_fn(WINDOW_DELAY_SECONDS)
            items_in_window = 0
            halfway = None
        if items_in_window >= ITEMS_PER_WINDOW:
            progress(f"Embedded {start}/{len(texts)}; "
                     f"waiting {WINDOW_DELAY_SECONDS} seconds for quota window...")
            sleep_fn(WINDOW_DELAY_SECONDS)
            items_in_window = 0
        batch_size = min(EMBEDDING_BATCH_SIZE, ITEMS_PER_WINDOW - items_in_window,
                         len(texts) - start,
                         halfway - start if halfway is not None else len(texts) - start)
        batch = texts[start:start + batch_size]
        retries = 0
        while True:
            try:
                result = embed_content(model=EMBEDDING_MODEL, content=batch,
                                       task_type="retrieval_document",
                                       request_options={"timeout": 60, "retry": None})
                break
            except Exception as error:
                delay = retry_delay(error, retries)
                if delay is None or retries >= MAX_RETRIES:
                    raise
                retries += 1
                progress(f"Embedding service unavailable; waiting {delay} seconds "
                         f"before retry {retries}/{MAX_RETRIES}...")
                sleep_fn(delay)
                items_in_window = 0
        vectors = np.asarray(result["embedding"], dtype=np.float32)
        if vectors.ndim == 1 and len(batch) == 1:
            vectors = vectors.reshape(1, -1)
        batches.append(validate_vectors(vectors, len(batch)))
        start += len(batch)
        items_in_window += len(batch)
        save_checkpoint(directory, source_bytes, len(texts), np.vstack(batches))
        progress(f"Embedded {start}/{len(texts)} curriculum sections")

    vectors = validate_vectors(np.vstack(batches), len(texts))
    if canonical_bytes((directory / DATASET_NAME).read_bytes()) != source_bytes:
        raise ValueError("Curriculum Markdown changed during generation; run again")
    metadata = embedding_metadata(source_bytes, len(texts))
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        binary = Path(temporary) / EMBEDDINGS_NAME
        meta = Path(temporary) / METADATA_NAME
        np.savez_compressed(binary, embeddings=vectors)
        metadata["embeddings_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
        meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        replace_with_retry(binary, directory / binary.name)
        replace_with_retry(meta, directory / meta.name)
    for name in (CHECKPOINT_BINARY, CHECKPOINT_METADATA):
        (directory / name).unlink(missing_ok=True)
    return len(texts)


def main():
    from dotenv import load_dotenv
    from gemini_service import create_client, embedding_function

    load_dotenv(APP_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY_INSURVERSE")
    if not api_key:
        print("Missing GEMINI_API_KEY_INSURVERSE in .env/environment")
        return 1
    client = create_client(api_key)
    try:
        count = build_embeddings(APP_DIR, embedding_function(client))
    except Exception as error:
        print(f"Curriculum embedding build failed: {error}")
        return 1
    finally:
        if client is not None:
            client.close()
    print(f"Saved {EMBEDDINGS_NAME} and {METADATA_NAME}: {count} sections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
