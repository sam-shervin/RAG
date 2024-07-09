from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import chromadb
import numpy as np

# Function to compute the cosine similarity between two vectors
def compute_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Load PDFs and split them into chunks
docs = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=300,
    length_function=len,
    is_separator_regex=False,
).split_documents(PyPDFDirectoryLoader("data").load())

# Create a ChromaDB collection and store document embeddings
client = chromadb.Client()
collection = client.create_collection(name="docs")
for d in docs:
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=d.page_content)["embedding"]
    collection.add(ids=[str(d.metadata)], embeddings=[embedding], documents=[d.page_content])

# Query the most relevant document for the given prompt
prompt = "How to install Astro?"
query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
results = collection.query(query_embeddings=[query_embedding], n_results=4)
context = str(results["documents"])

# Generate a response using the most relevant document context
output = ollama.generate(model="llama3-gradient", prompt=f"Using this data: {context}. Respond to this prompt: {prompt}")

print(output['response'])

#print("similarity: ",compute_cosine_similarity(np.array(response["embedding"]), np.array(results["embeddings"][0][0])))