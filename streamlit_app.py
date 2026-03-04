import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

st.set_page_config(page_title="SafeCare AI", page_icon="💬", layout="wide")

with st.sidebar:
    st.title("📘 SafeCare Assistant")
    st.markdown("### 🔍 RAG Based Chatbot")
    st.markdown("""
    - Model: Mistral-7B
    - Vector DB: FAISS
    - Embeddings: MiniLM
    """)
    st.divider()
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("💬 SafeCare – Medical Knowledge Retrieval System")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

db = load_vectorstore()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and "scores" in message:
            with st.expander("🔎 Retrieval Details"):
                st.write("**Chunk IDs:**", message["chunk_ids"])
                st.write("**Similarity Scores:**", message["scores"])

query = st.chat_input("Ask something about the document...")

if query:
    
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking... 🔎"):

            results = db.similarity_search_with_score(query, k=3)

            if not results:
                answer = "Not found in document."
                chunk_ids = "N/A"
                scores = "N/A"
            else:
                contexts = []
                chunk_ids = []
                scores = []

                for doc, score in results:
                    contexts.append(doc.page_content)
                    chunk_ids.append(doc.metadata.get("chunk_id"))
                    scores.append(round(score, 4))

                context = "\n\n".join(contexts)

                prompt = f"""
Use only the context below.
If answer is not found, say "Not found in document".

Context:
{context}

Question:
{query}
"""

                response = client.chat_completion(
                    messages=[
                        {"role": "system", "content": "You answer only using provided context."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=256,
                    temperature=0.2
                )

                answer = response.choices[0].message.content.strip()

            st.markdown(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "chunk_ids": chunk_ids,
            "scores": scores,
        }
    )

    st.rerun()