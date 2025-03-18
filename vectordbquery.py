import os
import numpy as np
import umap
import plotly.express as px
from datetime import datetime
import chromadb
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
collection = chroma_client.get_or_create_collection(name="my_collection")

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfReader(file)
        pages = [
            {"text": page.extract_text()} for page in reader.pages
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

    # Create chunks based on each page
    chunks = [page["text"] for page in pages if page["text"]]

    if not chunks:
        return "No valid text found in the PDF."

    # Generate embeddings
    embeddings = get_openai_embedding(chunks)

    # Add data to ChromaDB
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "source": pdf_path,
                "chunk_index": i,
                "date_added": datetime.now().isoformat(),
                "tags": "finance_annual_report,2022",
                "section": "General"}],
            ids=[f"doc_{i}"]
        )

    # Reduce dimensions using UMAP
    reducer = umap.UMAP(n_components=3)
    embeddings_3d = reducer.fit_transform(embeddings)

    # Create a 3D scatter plot
    fig = px.scatter_3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        title="3D Visualization of PDF Embeddings",
        hover_name=[f"Chunk {i}" for i in range(len(embeddings_3d))],
        color=np.arange(len(embeddings_3d)),
        color_continuous_scale='rainbow'
    )
    return fig

# Query data from ChromaDB
def query_documents():
    try:
        results = collection.query(n_results=5)
        if not results["documents"]:
            return "No data found in the collection."

        output = ""
        for i in range(len(results["documents"])):
            content = results["documents"][i]
            metadata = results["metadatas"][i]

            output += f"🔖 **Tags:** {metadata.get('tags', 'N/A')}\n"
            output += f"📅 **Date Added:** {metadata.get('date_added', 'Unknown')}\n"
            output += f"📚 **Section:** {metadata.get('section', 'General')}\n\n"
            output += f"📝 **Content:**\n{content}\n"
            output += "-" * 60 + "\n\n"

        return output
    except Exception as e:
        return f"❌ Query failed: {str(e)}"

# Gradio Interfaces
def upload_and_process(file):
    fig = process_pdf(file.name)
    return fig

def query_data():
    return query_documents()

# Combine Gradio interfaces using tabs
with gr.Blocks() as app:
    gr.Markdown("# 📘 PDF to ChromaDB with 3D Visualization")
    
    with gr.Tab("📤 Upload PDF"):
        gr.Interface(
            fn=upload_and_process,
            inputs=gr.File(label="Upload PDF"),
            outputs=gr.Plot(label="3D PDF Embeddings"),
            description="Upload a PDF to store its embeddings in ChromaDB and visualize them in 3D."
        )

    with gr.Tab("🔍 Query Data"):
        gr.Interface(
            fn=query_data,
            inputs=[],
            outputs=gr.Textbox(label="Stored Documents"),
            description="Retrieve stored documents and metadata from ChromaDB."
        )

app.launch(share=True)
