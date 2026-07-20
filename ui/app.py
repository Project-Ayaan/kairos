import os
import streamlit as st
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Kairos CDSS Testing UI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for high-quality dark styling
st.markdown("""
<style>
    /* Sleek container backgrounds */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Title styling */
    .app-title {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .app-subtitle {
        color: #8b949e;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Custom source cards */
    .source-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .source-card:hover {
        transform: translateY(-2px);
        border-color: #58a6ff;
    }
    
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    
    .source-title {
        font-weight: 700;
        color: #c9d1d9;
        font-size: 1.05rem;
    }
    
    .badge {
        font-size: 0.75rem;
        padding: 3px 8px;
        border-radius: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-pmid {
        background-color: #21262d;
        color: #58a6ff;
        border: 1px solid #30363d;
    }
    .badge-score {
        background-color: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
    }
    
    .source-meta {
        font-size: 0.85rem;
        color: #8b949e;
        margin-bottom: 8px;
    }
    
    .source-text {
        font-size: 0.92rem;
        color: #c9d1d9;
        line-height: 1.5;
        border-left: 3px solid #30363d;
        padding-left: 10px;
    }
    
    /* Status indicators */
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
    }
    .status-green { background-color: #2ea44f; }
    .status-red { background-color: #da3637; }
</style>
""", unsafe_allow_html=True)

# Imports from RAG pipeline
try:
    try:
        from ui.rag import embed_query, search_qdrant, generate_answer
    except ImportError:
        from rag import embed_query, search_qdrant, generate_answer
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    RAG_ERROR = e

# Sidebar Information
st.sidebar.markdown("<h2 style='text-align: center; color: #58a6ff;'>⚙️ System Controls</h2>", unsafe_allow_html=True)

# Qdrant status check
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "pubmed_articles"

qdrant_online = False
points_count = 0

try:
    client = QdrantClient(url=QDRANT_URL)
    # Check if we can fetch collections
    client.get_collections()
    qdrant_online = True
    if client.collection_exists(COLLECTION_NAME):
        info = client.get_collection(COLLECTION_NAME)
        points_count = info.points_count
except Exception:
    qdrant_online = False

if qdrant_online:
    st.sidebar.markdown(f'<p><span class="status-dot status-green"></span>Qdrant: <b>Online</b> ({QDRANT_URL})</p>', unsafe_allow_html=True)
    st.sidebar.metric(label="Ingested Chunks in DB", value=f"{points_count:,}")
else:
    st.sidebar.markdown(f'<p><span class="status-dot status-red"></span>Qdrant: <b>Offline</b> ({QDRANT_URL})</p>', unsafe_allow_html=True)

st.sidebar.markdown("---")

# RAG & Groq Settings
st.sidebar.markdown("### 🤖 Model & Retrieval Configuration")
groq_model_name = st.sidebar.text_input(
    "Groq Model ID",
    value="openai/gpt-oss-20b"
)
top_k = st.sidebar.slider("Top K Retrieved Chunks", min_value=1, max_value=10, value=5)
temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.8rem; color: #8b949e; text-align: center;">
    <b>Kairos CDSS</b> - Evidence-Grounded Clinical Decision Support System.<br>
    Built using MedCPT semantic embeddings, Qdrant vector database, and Groq API.
</div>
""", unsafe_allow_html=True)

# Main Application Layout
st.markdown('<h1 class="app-title">🩺 Kairos CDSS</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Evidence-Grounded Clinical Decision Support System & Verification Hub</p>', unsafe_allow_html=True)

if not RAG_AVAILABLE:
    st.error(f"Failed to import RAG pipeline components: {RAG_ERROR}")
    st.info("Ensure all requirements are satisfied by running: `uv sync --group etl --group ui`")
    st.stop()

# Sample queries for easy testing
st.markdown("### 💡 Quick Test Queries")
cols = st.columns(3)
queries = [
    "What is the role of clinical decision support systems in reducing medication errors?",
    "How does clinical decision support impact diagnostic accuracy and clinical outcomes?",
    "What are the barriers to adoption of clinical decision support systems in hospitals?"
]

selected_query = ""
for i, q in enumerate(queries):
    if cols[i].button(q, key=f"sample_q_{i}", use_container_width=True):
        selected_query = q

# Main text area for clinical query
query_input = st.text_area(
    "Enter Clinical Query:",
    value=selected_query if selected_query else "",
    placeholder="Type your clinical question or paste patient case details here to retrieve evidence-grounded insights...",
    height=100
)

# Ask Button
if st.button("🔍 Search & Synthesize Evidence", use_container_width=True):
    if not query_input.strip():
        st.warning("Please enter a valid clinical query.")
    elif not qdrant_online:
        st.error("Qdrant database is offline. Cannot query vector index.")
    elif points_count == 0:
        st.warning("The Qdrant collection is currently empty. Run the ETL pipeline script first: `python etl/run_etl.py`")
    else:
        with st.spinner("1. Generating query embedding via MedCPT-Query-Encoder..."):
            try:
                query_vector = embed_query(query_input)
            except Exception as e:
                st.error(f"Error generating query embedding: {e}")
                st.stop()
                
        with st.spinner("2. Searching Qdrant vector store (Dense Similarity)..."):
            try:
                retrieved_chunks = search_qdrant(query_vector, top_k=top_k)
            except Exception as e:
                st.error(f"Error searching Qdrant database: {e}")
                st.stop()
                
        if not retrieved_chunks:
            st.info("No relevant chunks found for the query.")
            st.stop()
            
        # Display Results
        ans_col, src_col = st.columns([3, 2])
        
        with ans_col:
            st.markdown("### 📋 Synthesized Clinical Answer")
            with st.spinner("3. Generating grounded response from Groq LLM..."):
                try:
                    answer = generate_answer(
                        query=query_input,
                        chunks=retrieved_chunks,
                        model_name=groq_model_name,
                        temperature=temperature
                    )
                    st.info(answer)
                except Exception as e:
                    st.error(f"Groq API Error: {e}")
                    st.write("Please verify that your `GROQ_API_KEY` is set correctly in `.env` and that the model ID is correct.")
                    
        with src_col:
            st.markdown("### 📚 Retrieved Evidence Sources")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">
                        <span class="source-title">[{idx}] {chunk['title']}</span>
                    </div>
                    <div class="source-meta">
                        <b>Authors:</b> {chunk['authors']}<br>
                        <b>Journal:</b> <i>{chunk['journal']}</i> ({chunk['year']})
                    </div>
                    <div style="margin-bottom: 8px;">
                        <span class="badge badge-pmid">PMID: {chunk['pmid']}</span>
                        <span class="badge badge-score">Cosine Similarity: {chunk['score']:.4f}</span>
                    </div>
                    <div class="source-text">
                        {chunk['text']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
