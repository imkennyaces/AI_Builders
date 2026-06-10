"""Build a FAISS vector index from ./docs/.

Run once after adding/changing documents:
    python ingest.py
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DOCS_DIR = Path("docs")
INDEX_DIR = Path("index")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": lambda p: TextLoader(p, encoding="utf-8"),
    ".md": UnstructuredMarkdownLoader,
}


def load_documents(folder: Path):
    docs = []
    for path in folder.rglob("*"):
        loader_cls = LOADERS.get(path.suffix.lower())
        if not loader_cls:
            continue
        print(f"  loading {path}")
        docs.extend(loader_cls(str(path)).load())
    return docs


def main() -> None:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.rglob("*")):
        raise SystemExit("Put some .pdf / .txt / .md files in ./docs first.")

    print("Loading documents...")
    raw_docs = load_documents(DOCS_DIR)
    print(f"  loaded {len(raw_docs)} document pages/sections")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  produced {len(chunks)} chunks")

    print(f"Embedding with {EMBED_MODEL} (downloads on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(INDEX_DIR))
    print(f"Saved index to ./{INDEX_DIR}/")


if __name__ == "__main__":
    main()