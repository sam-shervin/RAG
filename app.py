from langchain.document_loaders.pdf import PyPDFDirectoryLoader

# Load all PDFs in the data directory
pdf_loader = PyPDFDirectoryLoader("data").load()

