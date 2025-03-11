import os
import numpy as np
import umap
import chromadb
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from langchain.schema import Document
import PyPDF2
import plotly.express as px
import streamlit as st

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Initialize OpenAI and ChromaDB clients
client_openai = OpenAI()
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma_client.get_or_create_collection(name="my_collection")

MODEL = "text-embedding-ada-002"

def extract_text_from_pdf(pdf_file):
    pages = []
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text.strip():
            pages.append({
                "text": text,
                "page_number": page_num + 1
            })
    return pages

def get_openai_embedding(chunks):
    embeddings = []
    for chunk in chunks:
        response = client_openai.embeddings.create(
            input=chunk.page_content,
            model=MODEL
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# Streamlit App
st.title("PDF to ChromaDB with 3D Visualization")

pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf_file is not None:
    st.info("Processing the uploaded PDF...")

    # Extract text from PDF
    pages = extract_text_from_pdf(pdf_file)
    chunks = [Document(page_content=page["text"]) for page in pages]

    # Generate embeddings
    embeddings = get_openai_embedding(chunks)

    # Store data in ChromaDB
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk.page_content],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "source": pdf_file.name,
                "chunk_index": i,
                "page_number": pages[i]['page_number'],
                "date_added": datetime.now().isoformat(),
                "tags": "finance_annual_report,2022",
                "section": "General"}],
            ids=[f"doc_{i}"]
        )

    st.success("PDF successfully processed and stored in ChromaDB!")

    # Dimensionality reduction to 3D
    st.info("Generating 3D visualization...")
    reducer = umap.UMAP(n_components=3)
    embeddings_3d = reducer.fit_transform(embeddings)

    # Create random colors
    colors = np.random.randint(0, 10, len(embeddings_3d))

    # 3D Scatter plot
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        title="3D Visualization of PDF Embeddings",
        hover_name=[f"Page {pages[i]['page_number']}" for i in range(len(embeddings_3d))],
        color=colors,
        color_continuous_scale='rainbow'
    )

    st.plotly_chart(fig)

    st.balloons()