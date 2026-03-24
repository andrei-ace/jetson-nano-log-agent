"""Build FAISS index from field_manual.md. Run once offline, not on every agent launch."""

import json
import os

# Suppress C++ warnings from ONNX Runtime GPU discovery
_fd = os.dup(2)
os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
import onnxruntime as ort
ort.set_default_logger_severity(3)
from fastembed import TextEmbedding
os.dup2(_fd, 2)
os.close(_fd)

import faiss
import numpy as np

MANUAL_PATH = os.environ.get("KB_PATH", "./field_manual.md")
OUT_DIR = os.path.join(os.path.dirname(MANUAL_PATH), "kb_index")
MODEL_CACHE = os.path.join(OUT_DIR, "model_cache")
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Split markdown on ## headings. Returns (heading, body) pairs."""
    chunks = []
    current_heading = ""
    current_body: list[str] = []
    for line in text.split("\n"):
        if line.startswith("## "):
            if current_heading:
                chunks.append((current_heading, "\n".join(current_body).strip()))
            current_heading = line.removeprefix("## ").strip()
            current_body = []
        else:
            current_body.append(line)
    if current_heading:
        chunks.append((current_heading, "\n".join(current_body).strip()))
    return chunks


def main():
    with open(MANUAL_PATH) as f:
        text = f.read()

    chunks = chunk_markdown(text)
    print(f"Chunked {MANUAL_PATH} into {len(chunks)} sections")

    texts = [f"{heading}\n{body}" for heading, body in chunks]

    os.makedirs(MODEL_CACHE, exist_ok=True)
    # Skip network check if model already downloaded
    cached = any(
        f.endswith(".onnx")
        for root, _, files in os.walk(MODEL_CACHE)
        for f in files
    )
    print(f"Embedding with {MODEL_NAME}..." + (" (cached)" if cached else " (downloading)"))
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=MODEL_CACHE, local_files_only=cached)
    embeddings = np.array(list(model.embed(texts)), dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(OUT_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    with open(os.path.join(OUT_DIR, "chunks.json"), "w") as f:
        json.dump(chunks, f)

    print(f"Wrote {OUT_DIR}/index.faiss ({embeddings.shape[0]} vectors, dim={embeddings.shape[1]})")
    print(f"Wrote {OUT_DIR}/chunks.json ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
