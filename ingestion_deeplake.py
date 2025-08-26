# reinjest_deeplake_nodes.py
import os
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any

import deeplake
from dotenv import load_dotenv

# OpenAI + LlamaIndex
import openai
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.core.storage.storage_context import StorageContext

load_dotenv()
os.environ["ACTIVELOOP_TOKEN"] = os.getenv("ACTIVELOOP_TOKEN")
openai.api_key = os.getenv("OPENAI_API_KEY")
DATASET_NAME = "calum_v11"

AGENT_ID_DEFAULT = "1"

# ---------- Debug config ----------
DEBUG_SAMPLES_N = 5
DEBUG_OUT_PATH = "li_ingest_debug.jsonl"

class DebugRecorder:
    def __init__(self, n: int = 5, path: str = "li_ingest_debug.jsonl"):
        self.n = n
        self.path = path
        self.count = 0
        # truncate file each run
        try:
            open(self.path, "w", encoding="utf-8").close()
        except Exception:
            pass

    def record(self, nodes: List[TextNode]):
        if self.count >= self.n or not nodes:
            return
        to_take = min(self.n - self.count, len(nodes))
        samples = nodes[:to_take]
        with open(self.path, "a", encoding="utf-8") as f:
            for nd in samples:
                try:
                    node_json = nd.to_json()  # EXACT string LlamaIndex stores under metadata["node"]
                except Exception:
                    node_json = json.dumps(nd.to_dict())
                f.write(json.dumps({
                    "id": getattr(nd, "node_id", getattr(nd, "id_", None)),
                    "text_preview": (nd.text[:220].replace("\n", " ") if nd.text else None),
                    "text_len": len(nd.text or ""),
                    "metadata": nd.metadata,                 # your supplied metadata
                    "node_json_head": node_json[:240],       # preview (full string is long)
                }, ensure_ascii=False) + "\n")
        self.count += to_take

debug_recorder = DebugRecorder(DEBUG_SAMPLES_N, DEBUG_OUT_PATH)


class OpenAIDeepLakeIngestor:
    def __init__(self, dataset_name: str, overwrite: bool = True):
        token = os.getenv("ACTIVELOOP_TOKEN")
        org_id = os.getenv("ACTIVELOOP_ORG_ID")
        assert token and org_id, "Missing ACTIVELOOP credentials"

        self.ds_path = f"hub://{org_id}/{dataset_name}"

        # Configure LI query+ingest embedding model (same space)
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Let LlamaIndex own the vector store + metadata shape
        self.vector_store = DeepLakeVectorStore(
            dataset_path=self.ds_path,
            token=token,
            overwrite=overwrite,  # clean rebuild if True
        )
        self.storage = StorageContext.from_defaults(vector_store=self.vector_store)
        # Create a tiny index handle we can insert nodes into
        self.index = VectorStoreIndex([], storage_context=self.storage)

    # ----------- Utilities -----------
    def get_openai_embedding(self, text: str) -> List[float]:
        # still available for ad-hoc checks if you need it
        try:
            resp = openai.embeddings.create(
                model="text-embedding-3-small",
                input=[text],
            )
            return resp.data[0].embedding
        except Exception as e:
            print("OpenAI embedding failed:", e)
            return []

    def get_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        # kept for compatibility; LI handles embeddings on insert_nodes()
        if not texts:
            return []
        try:
            resp = openai.embeddings.create(
                model="text-embedding-3-small",
                input=texts,
            )
            return [d.embedding for d in resp.data]
        except Exception as e:
            print("OpenAI batch embedding failed:", e)
            return [self.get_openai_embedding(t) for t in texts]

    def chunk_text(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        if not text:
            return []

        def split_recursively(segment: str) -> List[str]:
            if len(segment) <= chunk_size:
                return [segment.strip()]
            paragraphs = re.split(r"\n\s*\n", segment)
            if len(paragraphs) > 1:
                chunks = []
                for p in paragraphs:
                    if p.strip():
                        chunks.extend(split_recursively(p))
                return chunks
            sentences = re.split(r"(?<=[.!?])\s+", segment)
            if len(sentences) > 1:
                chunks, current = [], ""
                for s in sentences:
                    if len(current) + len(s) + 1 <= chunk_size:
                        current = (current + " " + s).strip()
                    else:
                        if current:
                            chunks.extend(split_recursively(current))
                        current = s
                if current:
                    chunks.extend(split_recursively(current))
                return chunks
            words = segment.split()
            chunks, current = [], []
            for w in words:
                if sum(len(x) + 1 for x in current) + len(w) + 1 <= chunk_size:
                    current.append(w)
                else:
                    chunks.append(" ".join(current))
                    current = [w]
            if current:
                chunks.append(" ".join(current))
            return chunks

        raw = [c for c in split_recursively(text) if c.strip()]
        if not raw:
            return []

        out = []
        for i, c in enumerate(raw):
            if i == 0:
                out.append(c.strip())
            else:
                overlap_text = raw[i - 1][-overlap:] if overlap > 0 else ""
                out.append((overlap_text + c).strip())
        return [c for c in out if c]

    # ----------- Internal insert helper -----------
    def _insert_nodes(self, nodes: List[TextNode]):
        if not nodes:
            return
        # DEBUG: capture a few samples before writing
        debug_recorder.record(nodes)
        if debug_recorder.count <= DEBUG_SAMPLES_N:
            print("[DEBUG] sample:",
                  nodes[0].metadata.get("filename"),
                  "…",
                  (nodes[0].text[:100] or "").replace("\n", " "))
        # Write via LlamaIndex (embeds + stores proper metadata["node"])
        self.index.insert_nodes(nodes)

    # ----------- Ingestion (via LlamaIndex writer) -----------
    def _ingest_text(self, text: str, metadata_base: dict, batch_size: int = 64):
        """Normal text ingestion: uses your recursive+overlap chunker."""
        chunks = self.chunk_text(text)
        if not chunks:
            return

        for i in range(0, len(chunks), batch_size):
            block = chunks[i : i + batch_size]
            nodes = []
            for chunk in block:
                rid = str(uuid.uuid4())
                nodes.append(TextNode(id_=rid, text=chunk, metadata=dict(metadata_base)))
            self._insert_nodes(nodes)

    # ----------- QA ingestion (one node per Q+A pair) -----------
    def ingest_qa_chunks(self, qa_chunks: List[str], metadata_base: dict, batch_size: int = 64):
        """Each item in qa_chunks is treated as a single node; no extra chunking."""
        if not qa_chunks:
            return
        for i in range(0, len(qa_chunks), batch_size):
            block = qa_chunks[i : i + batch_size]
            nodes = []
            for chunk in block:
                rid = str(uuid.uuid4())
                nodes.append(TextNode(id_=rid, text=chunk, metadata=dict(metadata_base)))
            self._insert_nodes(nodes)

    def ingest_pdf(self, pdf_path: str, agent: str = AGENT_ID_DEFAULT):
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                full_text = "".join((page.extract_text() or "") for page in pdf.pages)

            meta = {
                "agent": agent,
                "timestamp": datetime.now().isoformat(),
                "filename": os.path.basename(pdf_path),
                "source_type": "pdf",
            }
            self._ingest_text(full_text, meta)
            print(f"Ingested PDF: {pdf_path}")
        except Exception as e:
            print(f"Failed to ingest {pdf_path}: {e}")

    def ingest_txt(self, txt_path: str, agent: str = AGENT_ID_DEFAULT):
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                full_text = f.read()

            meta = {
                "agent": agent,
                "timestamp": datetime.now().isoformat(),
                "filename": os.path.basename(txt_path),
                "source_type": "txt",
            }
            self._ingest_text(full_text, meta)
            print(f"Ingested TXT: {txt_path}")
        except Exception as e:
            print(f"Failed to ingest {txt_path}: {e}")

    # ----------- Q&A chunking (splitter) -----------
    def chunk_qa_pairs(self, text: str) -> List[str]:
        """
        Splits the questions file so each returned chunk is one Q+A block.
        Starts a new block on lines beginning with '#' or '**'.
        """
        qa_chunks, current_chunk = [], ""
        for line in text.splitlines():
            if re.match(r"^\s*(#|\*\*)", line):
                if current_chunk.strip():
                    qa_chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk.strip():
            qa_chunks.append(current_chunk.strip())
        print(f"Total Q&A chunks created: {len(qa_chunks)}")
        return qa_chunks


# ----------------- Usage -----------------
if __name__ == "__main__":
    AGENT_ID = "1"
    deeplake.delete(f"hub://{os.getenv("ACTIVELOOP_ORG_ID")}/calum_v11", force=True)

    ingestor = OpenAIDeepLakeIngestor(DATASET_NAME, overwrite=True)

    # 1) Plain text file(s) -> normal chunking
    txt_files = ["calum's brain.txt"]
    for file in txt_files:
        if os.path.exists(file):
            ingestor.ingest_txt(file, agent=AGENT_ID)
        else:
            print(f"Missing input: {file}")

    # 2) Q&A markdown ingestion -> ONE NODE PER QA PAIR
    qa_path = "calum's questions.txt"
    if os.path.exists(qa_path):
        with open(qa_path, "r", encoding="utf-8", errors="replace") as f:
            qa_text = f.read()
        qa_chunks = ingestor.chunk_qa_pairs(qa_text)  # each item = full Q+A block
        print(f"Ingesting {len(qa_chunks)} Q&A chunks (1 node per pair)...")
        meta = {
            "agent": AGENT_ID,
            "timestamp": datetime.now().isoformat(),
            "filename": os.path.basename(qa_path),
            "source_type": "qa_markdown",
        }
        ingestor.ingest_qa_chunks(qa_chunks, meta, batch_size=64)
    else:
        print(f"Missing input: {qa_path}")

    print(f"Re-ingestion complete via LlamaIndex writer. Debug file: {DEBUG_OUT_PATH}")

