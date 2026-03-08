Project Overview

This project implements a lightweight semantic search system using vector embeddings, fuzzy clustering, and a semantic caching mechanism. The system processes natural language queries and retrieves semantically similar documents from the 20 Newsgroups Dataset.

The system is designed to avoid redundant computations by caching previously processed queries using semantic similarity instead of exact string matching. The entire pipeline is exposed as an API using FastAPI.

System Architecture

Components

1. Embeddings

Documents are converted into vector representations using the all-MiniLM-L6-v2 model from Sentence Transformers.
This model provides efficient semantic embeddings suitable for similarity search.

2. Vector Database

All document embeddings are stored in a vector index built with FAISS.
FAISS allows efficient nearest-neighbor search over high-dimensional vectors.

3. Fuzzy Clustering

Documents are clustered using Gaussian Mixture Model.
Unlike hard clustering, this produces a probability distribution across clusters for each document, allowing documents to belong to multiple semantic topics.

This helps uncover overlapping semantic structures in the dataset.

4. Semantic Cache

A semantic cache stores previously processed queries and their results.

When a new query arrives:

The query is embedded.

It is compared with cached queries using Cosine Similarity.

If similarity exceeds a predefined threshold, the cached result is returned instead of recomputing the search.

This improves efficiency when users submit similar queries phrased differently.

5. FastAPI Service

The system is exposed through a REST API implemented using FastAPI.

Available endpoints:

POST /query

Accepts a natural language query and returns semantically similar documents.

Example request:

{
  "query": "space shuttle launch"
}

Example response:

{
  "query": "space shuttle launch",
  "result": [...]
}
GET /cache/stats

Returns cache statistics.

Example response:

{
  "total_entries": 5,
  "hit_count": 2,
  "miss_count": 3,
  "hit_rate": 0.4
}
DELETE /cache

Clears the semantic cache and resets statistics.

Installation

Clone the repository:

git clone <your-repository-link>
cd semantic-search-system

Install dependencies:

pip install -r requirements.txt
Running the Service

Start the API server:

uvicorn main:app

The server will start at:

http://127.0.0.1:8000

Interactive API documentation is available at:

http://127.0.0.1:8000/docs
Design Decisions

SentenceTransformer (all-MiniLM-L6-v2) was chosen because it provides strong semantic embeddings with relatively low computational cost.

FAISS was used for efficient large-scale vector similarity search.

Gaussian Mixture Model was used for fuzzy clustering since it provides probabilistic cluster membership rather than hard assignments.

Semantic caching was implemented using cosine similarity between query embeddings, allowing the system to detect semantically similar queries.

Future Improvements

Potential improvements include:

Using cluster information to optimize cache lookup.

Adding distributed caching for larger systems.

Deploying the service with Docker for easier scalability.

Conclusion

This project demonstrates a semantic search system that combines vector embeddings, fuzzy clustering, and semantic caching to efficiently process natural language queries and retrieve relevant documents.