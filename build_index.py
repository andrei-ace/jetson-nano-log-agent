"""Build FAISS index from all markdown docs. Run once offline, not on every agent launch."""

import glob
import json
import os

_devnull = os.open(os.devnull, os.O_WRONLY)
_saved = os.dup(2)
os.dup2(_devnull, 2)
os.close(_devnull)
import onnxruntime as ort
ort.set_default_logger_severity(3)
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
os.dup2(_saved, 2)
os.close(_saved)

import faiss
import numpy as np

DOCS_DIR = os.environ.get("DOCS_DIR", "./docs")
OUT_DIR = os.environ.get("KB_DIR", "./kb_index")
MODEL_CACHE = os.path.join(OUT_DIR, "model_cache")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


def chunk_markdown(text: str, source: str) -> list[tuple[str, str]]:
    """Split markdown on ## headings. Returns (heading, body) pairs with source tag."""
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
    md_files = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
    if not md_files:
        print(f"ERROR: No .md files found in {DOCS_DIR}")
        raise SystemExit(1)

    all_chunks = []
    for path in md_files:
        with open(path) as f:
            text = f.read()
        source = os.path.basename(path)
        chunks = chunk_markdown(text, source)
        print(f"  {source}: {len(chunks)} sections")
        all_chunks.extend(chunks)

    print(f"Total: {len(all_chunks)} sections from {len(md_files)} files")

    texts = [f"{heading}\n{body}" for heading, body in all_chunks]

    os.makedirs(MODEL_CACHE, exist_ok=True)
    embed_model = ensure_models()

    print("Embedding...")
    embeddings = np.array(list(embed_model.embed(texts)), dtype=np.float32)
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs(OUT_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    with open(os.path.join(OUT_DIR, "chunks.json"), "w") as f:
        json.dump(all_chunks, f)

    print(f"Wrote {OUT_DIR}/index.faiss ({embeddings.shape[0]} vectors, dim={embeddings.shape[1]})")
    print(f"Wrote {OUT_DIR}/chunks.json ({len(all_chunks)} chunks)")


def _is_cached(subdir_match: str = "") -> bool:
    """Check if any .onnx file exists in MODEL_CACHE (optionally matching subdir name)."""
    if not os.path.isdir(MODEL_CACHE):
        return False
    for root, _, files in os.walk(MODEL_CACHE):
        if subdir_match and subdir_match not in root.lower():
            continue
        if any(f.endswith(".onnx") for f in files):
            return True
    return False


def ensure_models():
    """Download embedding + re-ranker models if not cached. Returns embed model."""
    os.makedirs(MODEL_CACHE, exist_ok=True)

    embed_cached = _is_cached("bge")
    print(f"Embedder {EMBED_MODEL}: " + ("cached" if embed_cached else "downloading..."))
    embed_model = TextEmbedding(model_name=EMBED_MODEL, cache_dir=MODEL_CACHE,
                                local_files_only=embed_cached)

    rerank_cached = _is_cached("marco")
    print(f"Re-ranker {RERANK_MODEL}: " + ("cached" if rerank_cached else "downloading..."))
    TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=MODEL_CACHE,
                     local_files_only=rerank_cached)

    return embed_model


if __name__ == "__main__":
    import sys
    if "--models-only" in sys.argv:
        ensure_models()
    else:
        main()
