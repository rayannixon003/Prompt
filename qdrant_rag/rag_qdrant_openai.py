from openai import OpenAI
import uuid
import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Load environment variables

client = OpenAI(api_key=os.getenv("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"))

# Initialize embedding model and Qdrant vector DB
embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(":memory:")  # You can also connect to local or cloud Qdrant

collection_name = "rag_docs"

# Create Qdrant collection
qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Documents to embed
documents = [
    "Retrieval-Augmented Generation (RAG) combines vector search with generation models.",
    "Qdrant is a high-performance vector database for storing embeddings.",
    "The Eiffel Tower is located in Paris.",
    "OpenAI's GPT-3.5 can generate text based on prompts and context."
]

# Embed and insert into Qdrant
vectors = embedder.encode(documents).tolist()
points = [
    PointStruct(id=uuid.uuid4().int >> 64, vector=vec, payload={"text": doc})
    for vec, doc in zip(vectors, documents)
]
qdrant.upsert(collection_name=collection_name, points=points)

# User query
query = "What is RAG and how does it work?"
query_vector = embedder.encode(query).tolist()

# Search similar docs
results = qdrant.search(
    collection_name=collection_name,
    query_vector=query_vector,
    limit=3
)

# Build prompt with context
retrieved_docs = [hit.payload["text"] for hit in results]
context = "\n".join(retrieved_docs)

# Generate answer using OpenAI's new SDK (v1.0+)
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"}
    ],
    temperature=0.7,
    max_tokens=200
)

print("=== RAG Answer ===")
print(response.choices[0].message.content)
