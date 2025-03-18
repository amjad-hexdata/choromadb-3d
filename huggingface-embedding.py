import os
import numpy as np
import umap
import plotly.express as px
from datetime import datetime
import chromadb
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Initialize Hugging Face Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB Client
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="HEXDATA-COLLECTION")

# Extract text from PDF and retain page numbers
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        pages = [
            {"text": page.extract_text(), "page_number": i + 1}
            for i, page in enumerate(reader.pages)
            if page.extract_text()  # Ensure non-empty text
        ]
    return pages

# Function to get embeddings using Hugging Face
def get_huggingface_embedding(chunks):
    return embedding_model.encode(chunks).tolist()  # Convert to list

# Process PDF and store in ChromaDB
def process_pdf(pdf_path):
    pages = extract_text_from_pdf(pdf_path)

    if not pages:
        return "No valid text found in the PDF."

    # Extract text from pages
    chunks = [page["text"] for page in pages]

    # Generate embeddings
    embeddings = get_huggingface_embedding(chunks)

    # Add data to ChromaDB with metadata
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],  # Directly store embedding
            metadatas=[{
                "source": pdf_path,
                "chunk_index": i,
                "page_number": pages[i]['page_number'],
                "date_added": datetime.now().isoformat(),
                "tags": "finance_annual_report,2022",
                "section": "General"
            }],
            ids=[f"doc_{pdf_path}_page_{i}"]
        )

    # Reduce dimensions using UMAP for visualization
    reducer = umap.UMAP(n_components=3)
    embeddings_3d = reducer.fit_transform(np.array(embeddings))

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