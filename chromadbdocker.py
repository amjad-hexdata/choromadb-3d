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
from sklearn.manifold import TSNE

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

pdf_path = "angroreport2022.pdf"
text = extract_text_from_pdf(pdf_path)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunks = text_splitter.create_documents([text])

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
        metadatas=[{"source": pdf_path}],
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