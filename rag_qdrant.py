from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import uuid


embedder = SentenceTransformer("all-MiniLM-L6-v2")
generator = pipeline("text-generation", model="gpt2")


client = QdrantClient(":memory:")
collection_name = "my_docs"


client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)


documents = [
    "RAG means retrieval-augmented generation.",
    "Qdrant is a vector database for similarity search.",
    "The Eiffel Tower is in Paris.",
    "Python is popular for AI tasks.",
]


embeddings = embedder.encode(documents).tolist()
points = [
    PointStruct(id=uuid.uuid4().int >> 64, vector=vec, payload={"text": doc})
    for vec, doc in zip(embeddings, documents)
]
client.upsert(collection_name=collection_name, points=points)


query = "What is RAG?"
query_vec = embedder.encode(query).tolist()
hits = client.search(collection_name, query_vector=query_vec, limit=3)


retrieved_docs = [hit.payload["text"] for hit in hits]
context = "\n".join(retrieved_docs)


prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
output = generator(prompt, max_length=100, do_sample=True)[0]["generated_text"]
print(output)
