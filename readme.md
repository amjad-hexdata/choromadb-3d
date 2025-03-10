# PDF Text Extractor and Embedding Visualizer

This project extracts text from a PDF file, generates embeddings using OpenAI, stores them in ChromaDB, and visualizes the embeddings in 3D using UMAP and Plotly.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The project performs the following steps:
1. Extracts text from a PDF file.
2. Splits the text into chunks.
3. Generates embeddings for each chunk using OpenAI's `text-embedding-ada-002` model.
4. Stores the embeddings in a ChromaDB collection.
5. Reduces the dimensionality of the embeddings to 3D using UMAP.
6. Visualizes the embeddings in an interactive 3D scatter plot using Plotly.

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
   git clone https://github.com/amjad-hexdata/choromadb-3d.git
   cd your-repo-name

  
   Contact
For questions or feedback, please contact:

Your Name: Muhammad Amjad

Email: m.amjad@hexdata.co.jp

GitHub: amjad-hexdata