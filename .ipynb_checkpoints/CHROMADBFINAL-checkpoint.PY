import os
import numpy as np
import umap
import plotly.express as px
from datetime import datetime
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import gradio as gr

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Initialize OpenAI and ChromaDB clients
client_openai = OpenAI()
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="HEXDATA-COLLECTION")

# Extract text from PDF and retain page numbers
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        pages = [
            {"text": page.extract_text(), "page_number": i + 1}
            for i, page in enumerate(reader.pages)
        ]
    return pages

# Function to get embeddings from OpenAI
def get_openai_embedding(chunks):
    embeddings = []
    for chunk in chunks:
        response = client_openai.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# Process PDF and store in ChromaDB
def process_pdf(pdf_path):
    pages = extract_text_from_pdf(pdf_path)

    # Create chunks based on each page to preserve page number
    chunks = [page["text"] for page in pages if page["text"]]

    # Generate embeddings
    embeddings = get_openai_embedding(chunks)

    # Add data to ChromaDB with actual page numbers
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "source": pdf_path,
                "chunk_index": i,
                "page_number": pages[i]['page_number'],
                "date_added": datetime.now().isoformat(),
                "tags": "finance_annual_report,2022",
                "section": "General"}],
            ids=[f"doc_{i}"]
        )

    # Reduce dimensions using UMAP
    reducer = umap.UMAP(n_components=3)
    embeddings_3d = reducer.fit_transform(embeddings)

    # Generate random colors
    colors = np.random.randint(0, 10, len(embeddings_3d))

    # Create a 3D scatter plot
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        title="3D Visualization of PDF Embeddings",
        hover_name=[f"Page {pages[i]['page_number']}" for i in range(len(embeddings_3d))],
        color=colors,
        color_continuous_scale='rainbow'
    )
    return fig

# Gradio web app
def upload_and_process(file):
    fig = process_pdf(file.name)
    return fig

iface = gr.Interface(
    fn=upload_and_process,
    inputs=gr.File(label="Upload PDF"),
    outputs=gr.Plot(label="3D PDF Embeddings"),
    title="PDF to ChromaDB with 3D Visualization",
    description="Upload a PDF file, store its embeddings in ChromaDB, and visualize them in 3D."
)

iface.launch(share=True)