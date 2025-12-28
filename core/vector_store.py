from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle, os

EMBED = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "data/index.faiss"
META_PATH  = "data/meta.pkl"

def build_index(papers: list[dict], index_path=INDEX_PATH, meta_path=META_PATH):
    texts = [f"{p['title']}\n{p['summary']}" for p in papers]
    meta  = [{"title": p["title"], "url": p["url"], "updated": str(p["updated"])} for p in papers]

    model = SentenceTransformer(EMBED)
    emb = model.encode(texts, normalize_embeddings=True).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"texts": texts, "meta": meta}, f)

def load_index(index_path=INDEX_PATH, meta_path=META_PATH):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        store = pickle.load(f)
    return index, store

def search_topk(query: str, k=5, index_path=INDEX_PATH, meta_path=META_PATH):
    index, store = load_index(index_path, meta_path)
    model = SentenceTransformer(EMBED)
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q, k)
    idx = idx[0].tolist()
    return [(store["texts"][i], store["meta"][i], float(scores[0][n])) for n,i in enumerate(idx)]
