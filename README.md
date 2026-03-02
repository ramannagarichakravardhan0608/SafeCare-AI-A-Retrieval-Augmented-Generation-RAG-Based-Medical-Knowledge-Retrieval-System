🏥 SafeCare-Medical-RAG-Assistant
An AI-powered Medical Document Assistant built using Retrieval-Augmented Generation (RAG).
This system retrieves relevant medical document chunks using FAISS and generates accurate answers using Mistral-7B — strictly based on the provided context.

🚀 Project Overview

SafeCare is a healthcare-focused chatbot that:

Reads medical PDF documents
Splits text into meaningful chunks
Assigns unique chunk_id to each section
Generates embeddings using MiniLM
Stores vectors in FAISS
Retrieves top similar chunks
Sends context to Mistral-7B LLM
Displays similarity scores + chunk IDs for transparency
The model answers only from retrieved document content.

🏗️ Architecture
Document (PDF)
        ↓
Text Splitter
        ↓
Add chunk_id
        ↓
Embeddings (MiniLM)
        ↓
FAISS Vector Store
        ↓
Similarity Search (Top K)
        ↓
Send Context to Mistral-7B
        ↓
Answer + Show chunk_id + Scores

📂 Project Structure

SafeCare-Medical-RAG-Assistant/
│
├── app.py                   # Streamlit Chat Application
├── create_vectorstore.py    # PDF Processing & FAISS Creation
├── architecture.txt         # System Flow Architecture
├── requirements.txt         # Project Dependencies
├── report.pdf               # Medical Document (Input)
└── vectorstore/             # Saved FAISS Database


🧠 Tech Stack

LLM: Mistral-7B-Instruct (via HuggingFace Inference API)

Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector Database: FAISS
Framework: Streamlit
Orchestration: LangChain
PDF Loader: PyPDFLoader


Installation & Setup
1 Clone Repository

git clone https://github.com/your-username/SafeCare-Medical-RAG-Assistant.git
cd SafeCare-Medical-RAG-Assistant

2 Install Dependencies
pip install -r requirements.txt

3 Add Environment Variable
HUGGINGFACEHUB_API_TOKEN=your_token_here

4 Create Vector Store
python create_vectorstore.py

5 Run Streamlit App
streamlit run app.py

Features

Chat-based Medical Document Q&A
Retrieval-Augmented Generation (RAG)
Shows Similarity Scores
Displays Retrieved Chunk IDs
Transparent Context-Based Answers
Clear Chat Option
Modern Streamlit Chat UI

Safety Design

The assistant:
Uses only retrieved document context
Returns "Not found in document." if answer not available
Prevents hallucinated responses

Example Workflow

User asks a medical question
FAISS retrieves top 3 relevant chunks
System sends context to Mistral-7B
Model generates answer
Chunk IDs + similarity scores displayed


Requirements

Key dependencies:

langchain
langchain-community
langchain-huggingface
sentence-transformers
faiss-cpu
streamlit
transformers
torch
pypdf

Use Cases

Medical Research Assistance
Clinical Document Querying
Hospital Knowledge Base Systems
Healthcare AI Demonstration Projects
RAG Architecture Learning

Future Improvements

Multiple PDF Upload Support
Similarity Score Visualization
Deployment on Streamlit Cloud
Role-Based Access
Multi-Document Retrieval

Author

Chakravardhan
Aspiring AI & Full Stack Developer
