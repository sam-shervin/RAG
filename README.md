# RAG with LLaMA3-Gradient Large Language Model(LLM) 

This project demonstrates the integration of advanced NLP techniques and machine learning models to process documents, compute their embeddings, and generate responses to user queries. It leverages the `LLaMA3-Gradient` model for generating responses and uses document embeddings to find the most relevant information. The project utilizes libraries such as `ollama`, `chromadb`, and `numpy`, along with custom document loading and text splitting functionalities.

## Features

- **Document Processing**: Load and split PDF documents into manageable chunks for further processing.
- **Embedding Generation**: Utilize the `ollama` library to generate embeddings for each document chunk, facilitating the measurement of document similarity.
- **ChromaDB Integration**: Store and manage document embeddings in a `ChromaDB` collection, enabling efficient querying of documents based on cosine similarity to a user query.
- **Query Processing**: Embed user queries and retrieve the most relevant document chunks from the `ChromaDB` collection.
- **Response Generation**: Generate responses to user queries based on the content of the most relevant document chunks using the `LLaMA3-Gradient` model.

## Setup

To set up this project, ensure you have Python installed and then install the required libraries:

```bash
pip install numpy ollama chromadb langchain_text_splitters langchain_community
```

Place your collection of PDF documents in a directory named `data` for the document loader to process.

## Usage

1. **Document Processing**: Run the script to load PDF documents from the `data` directory, split them into chunks, and generate embeddings for each chunk.
2. **ChromaDB Collection**: The script automatically creates a `ChromaDB` collection and stores the document embeddings.
3. **Query Processing**: Modify the `prompt` variable in the script to your specific query.
4. **Response Generation**: The script generates a response based on the most relevant document content and prints it to the console.

Example query modification:

```python
prompt = "Your specific query here"
```
