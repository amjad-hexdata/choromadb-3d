import streamlit as st
import chromadb
from PyPDF2 import PdfReader
from datetime import datetime

# Initialize ChromaDB Client (Docker-based)
client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_or_create_collection(name="HEXDATA-COLLECTION")

# Function to Extract Text from Each Page
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    pages = [
        {"text": page.extract_text(), "page_number": i + 1}
        for i, page in enumerate(reader.pages)
        if page.extract_text()
    ]
    return pages

# Streamlit UI
st.title("📄 Upload PDF to ChromaDB (Page by Page)")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    st.success("✅ PDF uploaded successfully!")

    # Extract text by pages
    pages = extract_text_from_pdf(uploaded_file)

    if pages:
        for page in pages:
            collection.add(
                documents=[page["text"]],
                metadatas=[{
                    "source": uploaded_file.name,
                    "page_number": page["page_number"],
                    "date_added": datetime.now().isoformat()
                }],
                ids=[f"{uploaded_file.name}_page_{page['page_number']}"]
            )

        st.success(f"✅ {len(pages)} pages stored in ChromaDB!")

# Display Collection Data
if st.button("Show Collection Data"):
    stored_data = collection.get()
    st.json(stored_data)
