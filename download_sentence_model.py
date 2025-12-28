from sentence_transformers import SentenceTransformer

print("Downloading all-MiniLM-L6-v2 from HuggingFace...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Download finished and cached locally.")
