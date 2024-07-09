from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the PDFs from the data directory
pdf_loader = PyPDFDirectoryLoader("data").load()

# Split the PDFs into chunks of 800 characters with an overlap of 80 characters
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )
print(text_splitter.split_documents(pdf_loader))
