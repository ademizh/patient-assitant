"""Persistent Chroma vector store for MEDICAL DOCUMENTS ONLY.

Patient intake data is never written here. Collections are namespaced by the embedding
model, so demo (hash) vectors and OpenAI vectors are never mixed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..text_utils import detect_language
from .chunking import chunk_document
from .loaders import LoadedDocument, load_document


@dataclass
class IndexResult:
    filename: str
    title: str
    doc_hash: str
    status: str  # "indexed" | "duplicate"
    n_chunks: int


@dataclass
class DocumentInfo:
    doc_hash: str
    title: str
    filename: str
    n_chunks: int
    pages: int | None
    chunk_size: int
    chunk_overlap: int


def _collection_name(prefix: str, signature: str) -> str:
    raw = f"{prefix}__{signature}"
    name = re.sub(r"[^A-Za-z0-9._-]", "-", raw)[:60].strip("-._")
    return name if len(name) >= 3 else f"{name}doc"


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name).name)[:120]


class DocumentStore:
    def __init__(self, chroma_path: Path, library_path: Path, prefix: str, embedder):
        """`embedder` is any object with `.embed(texts)` and `.embedding_signature`."""
        self.embedder = embedder
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(chroma_path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection_name = _collection_name(prefix, embedder.embedding_signature)
        self.collection = self.client.get_or_create_collection(
            self.collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding": embedder.embedding_signature, "content": "medical_documents"},
        )

    # ---- indexing ----------------------------------------------------------------------

    def is_indexed(self, doc_hash: str) -> bool:
        return bool(self.collection.get(where={"doc_hash": doc_hash}, limit=1, include=[])["ids"])

    def _save_original(self, doc: LoadedDocument, data: bytes) -> None:
        target = self.library_path / f"{doc.doc_hash[:16]}__{_safe_filename(doc.filename)}"
        if not target.exists():
            target.write_bytes(data)

    def add_file(self, filename: str, data: bytes, chunk_size: int, chunk_overlap: int) -> IndexResult:
        doc = load_document(filename, data)
        if self.is_indexed(doc.doc_hash):
            self._save_original(doc, data)
            return IndexResult(doc.filename, doc.title, doc.doc_hash, "duplicate", 0)
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata(chunk_size, chunk_overlap) for c in chunks],
        )
        self._save_original(doc, data)
        return IndexResult(doc.filename, doc.title, doc.doc_hash, "indexed", len(chunks))

    def delete_document(self, doc_hash: str, remove_file: bool = True) -> None:
        self.collection.delete(where={"doc_hash": doc_hash})
        if remove_file:
            for f in self.library_path.glob(f"{doc_hash[:16]}__*"):
                f.unlink(missing_ok=True)

    def stored_files(self) -> list[Path]:
        return sorted(p for p in self.library_path.glob("*__*") if p.is_file())

    def unindexed_files(self) -> list[Path]:
        indexed = {d.doc_hash[:16] for d in self.list_documents()}
        return [p for p in self.stored_files() if p.name.split("__", 1)[0] not in indexed]

    def reindex_all(self, chunk_size: int, chunk_overlap: int) -> list[IndexResult]:
        results = []
        for path in self.stored_files():
            original_name = path.name.split("__", 1)[1]
            data = path.read_bytes()
            doc = load_document(original_name, data)
            self.collection.delete(where={"doc_hash": doc.doc_hash})
            results.append(self.add_file(original_name, data, chunk_size, chunk_overlap))
        return results

    # ---- listing -----------------------------------------------------------------------

    def count(self) -> int:
        return self.collection.count()

    def list_documents(self) -> list[DocumentInfo]:
        if self.collection.count() == 0:
            return []
        got = self.collection.get(include=["metadatas"])
        docs: dict[str, DocumentInfo] = {}
        pages: dict[str, set[int]] = {}
        for md in got["metadatas"]:
            h = md["doc_hash"]
            if h not in docs:
                docs[h] = DocumentInfo(h, md["title"], md["filename"], 0, None, md["chunk_size"], md["chunk_overlap"])
                pages[h] = set()
            docs[h].n_chunks += 1
            if md.get("page", -1) and md.get("page", -1) > 0:
                pages[h].add(md["page"])
        for h, d in docs.items():
            d.pages = len(pages[h]) or None
        return sorted(docs.values(), key=lambda d: d.title.lower())

    # ---- retrieval ---------------------------------------------------------------------

    def query(self, question: str, top_k: int) -> list[dict]:
        n = self.collection.count()
        if n == 0:
            return []
        q_emb = self.embedder.embed([question])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=min(top_k, n),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, text, md, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            page = md.get("page", -1)
            out.append(
                {
                    "chunk_id": cid,
                    "text": text,
                    "title": md.get("title", ""),
                    "filename": md.get("filename", ""),
                    "page": page if page and page > 0 else None,
                    "heading": md.get("heading") or None,
                    "doc_hash": md.get("doc_hash"),
                    "score": round(1.0 - float(dist), 4),  # cosine similarity
                    "language": md.get("language") or detect_language(text),
                }
            )
        return out

    def query_multi(self, queries: list[str], top_k: int) -> list[dict]:
        """Retrieve for several phrasings (e.g. Russian + English) and keep the best score per chunk."""
        best: dict[str, dict] = {}
        for q in dict.fromkeys(q for q in queries if q and q.strip()):
            for c in self.query(q, top_k):
                if c["chunk_id"] not in best or c["score"] > best[c["chunk_id"]]["score"]:
                    best[c["chunk_id"]] = c
        return sorted(best.values(), key=lambda c: -c["score"])[:top_k]
