import os
import glob
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import PyPDF2
import umap
import numpy as np
import plotly.express as px
from datetime import datetime
import streamlit as st

def extract_text_from_pdf(pdf_path):

    with open(pdf_path, "rb") as file:
        # Initialize a PDF reader object
        reader = PyPDF2.PdfReader(file)
        
        # Initialize an empty list to store page data
        pages = []
        
        # Iterate over each page in the PDF
        for page_num, page in enumerate(reader.pages):
            # Extract text from the current page
            text = page.extract_text()
            
            # Append the text and page number to the list
            pages.append({
                "text": text,  # Text extracted from the page
                "page_number": page_num + 1  # Page number (starting from 1)
            })
    
    # Return the list of pages with text and page numbers
    return pages
pdf_path = "angroreport2022.pdf"
pages = extract_text_from_pdf(pdf_path)

#text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
#chunks = text_splitter.create_documents([text])
chunks = [Document(page_content=page["text"]) for page in pages]

MODEL = "text-embedding-ada-002"
db_name = "vector_db"

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

client_openai = OpenAI()

# Function to get embeddings from OpenAI
def get_openai_embedding(chunks):
    embeddings = []
    for chunk in chunks:
        response = client_openai.embeddings.create(
            input=chunk.page_content,
            model=MODEL
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# Generate embeddings for the extracted text
embeddings = get_openai_embedding(chunks)

# Initialize ChromaDB client
# client = chromadb.PersistentClient(path=db_name)
chroma_client = chromadb.HttpClient(host="localhost", port = 8000)
collection = chroma_client.get_or_create_collection(name="my_collection")
print("Collection created or retrieved:", collection.name)

# Add the embeddings to the collection
for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
    collection.add(
        documents=[chunk.page_content],
        embeddings=[embedding.tolist()],
        metadatas=[{
            "source": pdf_path,
            "chunk_index": i,
            "page_number": pages[i]['page_number'],  # Approximate page number
            "date_added": datetime.now().isoformat(),
            "tags": "finance_annual_report,2022",
            "section": "General"}],
        ids=[f"doc_{i}"]
    )
    
# Reduce dimensions to 3D using UMAP
reducer = umap.UMAP(n_components=3)  # Keep `random_state` for reproducibility if needed
embeddings_3d = reducer.fit_transform(embeddings)

# Create a 3D scatter plot
fig = px.scatter_3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    title="3D Visualization of PDF Embeddings",
    hover_name=[f"Point {i}" for i in range(len(embeddings_3d))]
)

# Show the plot
fig.show()