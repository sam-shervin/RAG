from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
import chromadb
import numpy as np


# Function to compute the cosine similarity between two vectors
def compute_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v))


# Load the PDFs from the data directory
pdf_loader = PyPDFDirectoryLoader("data").load()

# Split the PDFs into chunks of 800 characters with an overlap of 80 characters
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
docs = text_splitter.split_documents(pdf_loader)

client = chromadb.Client()
collection = client.create_collection(name="docs")

# store each document in a vector embedding database
for i, d in enumerate(docs):
  response = ollama.embeddings(model="nomic-embed-text", prompt=d.page_content)
  embedding = response["embedding"]
  collection.add(
    ids=[str(d.metadata)],
    embeddings=[embedding],
    documents=[d.page_content]
  )


prompt = "How to install Astro?"

# generate an embedding for the prompt and retrieve the most relevant doc
response = ollama.embeddings(
  prompt=prompt,
  model="nomic-embed-text"
)
results = collection.query(
  query_embeddings=[response["embedding"]],
  n_results=4,
)

#print("similarity: ",compute_cosine_similarity(np.array(response["embedding"]), np.array(results["embeddings"][0][0])))

context = str(results["documents"])
# get the most relevant document and generate a response
output = ollama.generate(
  model="llama3-gradient",
  prompt=f"Using this data: {context}. Respond to this prompt: {prompt}"
)

print(output['response'])
