# 3D Visualization of PDF Embeddings using ChromaDB and OpenAI

This project demonstrates how to extract text from a PDF, generate embeddings using OpenAI, store them in ChromaDB, and visualize the embeddings in 3D using UMAP and Plotly.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
6. [Contributing](#contributing)
7. [License](#license)

---

## Overview
The project performs the following steps:
1. Extracts text from a PDF file.
2. Splits the text into smaller chunks.
3. Generates embeddings for each chunk using OpenAI's `text-embedding-ada-002` model.
4. Stores the embeddings in a ChromaDB vector database.
5. Reduces the dimensionality of the embeddings to 3D using UMAP.
6. Visualizes the 3D embeddings using Plotly.

---

## Features
- **Text Extraction**: Extracts text from PDF files using `PyPDF2`.
- **Text Splitting**: Splits text into smaller chunks using `langchain-text-splitters`.
- **Embedding Generation**: Generates embeddings using OpenAI's API.
- **Vector Database**: Stores embeddings in ChromaDB for efficient retrieval.
- **3D Visualization**: Reduces embeddings to 3D using UMAP and visualizes them using Plotly.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key (set in `.env` file or environment variables)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/amjad_hexdata/choromadb-3d.git
   cd your-repo-name
   
   
   Contact
For questions or feedback, please contact:

Your Name

Email: m.amjad@hexdata.co.jp

GitHub: amjad-hexdata