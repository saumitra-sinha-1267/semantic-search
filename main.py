# ---------------------------
# Imports
# ---------------------------
from sklearn.datasets import fetch_20newsgroups
from sklearn.mixture import GaussianMixture
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

import numpy as np
import faiss

app = FastAPI()
class QueryRequest(BaseModel):
    query: str
# ---------------------------
# Load Dataset
# ---------------------------
data = fetch_20newsgroups(
    subset='all',
    remove=('headers', 'footers', 'quotes')
)

documents = data.data

print("Number of documents:", len(documents))

# ---------------------------
# Load Embedding Model
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Generate Embeddings
# ---------------------------
print("Generating embeddings...")

embeddings = model.encode(documents)

print("Embedding shape:", embeddings.shape)
import numpy as np

np.save("embeddings.npy", embeddings)

# ---------------------------
# FAISS Vector Index
# ---------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

embeddings = embeddings.astype("float32")

index.add(embeddings)

print("Total vectors in index:", index.ntotal)

# ---------------------------
# Fuzzy Clustering (GMM)
# ---------------------------
n_clusters = 20

gmm = GaussianMixture(
    n_components=n_clusters,
    covariance_type="full",
    random_state=42
)

gmm.fit(embeddings)

cluster_probs = gmm.predict_proba(embeddings)

print("Cluster probability shape:", cluster_probs.shape)

dominant_clusters = np.argmax(cluster_probs, axis=1)

print("Example cluster distribution:", cluster_probs[0])

# ---------------------------
# Semantic Cache
# ---------------------------
semantic_cache = []

hit_count = 0
miss_count = 0

SIMILARITY_THRESHOLD = 0.9

# ---------------------------
# Query Processing Function
# ---------------------------
def process_query(query):

    global hit_count, miss_count

    # Generate embedding for query
    query_embedding = model.encode([query])

    # Check cache
    for entry in semantic_cache:

        similarity = util.cos_sim(query_embedding, entry["embedding"]).item()

        print("Similarity:", similarity)

        if similarity > SIMILARITY_THRESHOLD:
            hit_count += 1
            print("CACHE HIT")
            return entry["result"]

    # Cache miss
    miss_count += 1

    query_embedding_faiss = query_embedding.astype("float32")

    k = 5

    distances, indices = index.search(query_embedding_faiss, k)

    results = [documents[i] for i in indices[0]]

    semantic_cache.append({
        "query": query,
        "embedding": query_embedding,
        "result": results
    })

    print("CACHE MISS")

    return results
@app.post("/query")
def query_api(request: QueryRequest):

    results = process_query(request.query)

    return {
        "query": request.query,
        "result": results
    }

@app.get("/cache/stats")
def cache_stats():

    total_entries = len(semantic_cache)
    total_queries = hit_count + miss_count

    hit_rate = hit_count / total_queries if total_queries else 0

    return {
        "total_entries": total_entries,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": hit_rate
    }

@app.delete("/cache")
def clear_cache():

    global semantic_cache, hit_count, miss_count

    semantic_cache = []
    hit_count = 0
    miss_count = 0

    return {"message": "Cache cleared"}


