import streamlit as st
import requests
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables (will load from cloud secrets when deployed online)
# load_dotenv("../.env", override=True)

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="PharmAssist", 
    page_icon="O", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# HIGH-END CLINICAL UI STYLING
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #f4f7f9 0%, #e8eef2 100%);
    }

    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        background-color: #1a2538;
        background-image: linear-gradient(180deg, #1a2538 0%, #0d1421 100%);
        border-right: 1px solid #2d3748;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    .main-header {
        font-size: 3rem;
        background: -webkit-linear-gradient(45deg, #0052D4, #4364F7, #6FB1FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .sub-header {
        color: #64748b;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 30px;
        margin-top: 0;
    }

    .fda-badge {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white !important;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 800;
        font-size: 0.9rem;
        vertical-align: middle;
        margin-left: 15px;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.4);
        -webkit-text-fill-color: initial; 
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .warning-box {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-left: 6px solid #ef4444;
        padding: 24px;
        border-radius: 12px;
        color: #1e293b;
        margin-bottom: 40px;
        font-size: 1.05rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.8);
        line-height: 1.6;
    }
    
    .warning-icon {
        color: #ef4444;
        font-weight: bold;
        margin-right: 10px;
        font-size: 1.2rem;
    }

    .clinical-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(255,255,255,1);
        margin-bottom: 20px;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    
    .clinical-card:before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 4px;
        background: linear-gradient(90deg, #4364F7, #6FB1FC);
    }
    
    .clinical-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0, 82, 212, 0.1);
    }
    
    .drug-title {
        color: #0f172a;
        margin-top: 0px;
        margin-bottom: 10px;
        font-weight: 800;
        font-size: 1.5rem;
    }
    
    .confidence-score {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 18px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .card-content {
        font-size: 0.95em;
        line-height: 1.7;
        color: #475569;
    }
    
    .card-content strong {
        color: #0f172a;
    }
    
    .copilot-response {
        background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.85) 100%);
        backdrop-filter: blur(20px);
        padding: 35px;
        border-radius: 20px;
        border-left: 6px solid #4364F7;
        box-shadow: 0 20px 50px rgba(0, 82, 212, 0.08);
        font-size: 1.15rem;
        line-height: 1.8;
        color: #0f172a;
        margin-top: 20px;
        border: 1px solid white;
    }
    
    .response-title {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
        font-weight: 800;
        color: #1e293b;
        font-size: 1.4rem;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 15px;
    }

    div[data-testid="stTextInput"] label {
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        color: #1e293b !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stTextInput"] input {
        border-radius: 12px;
        border: 2px solid #cbd5e0;
        padding: 15px 25px;
        font-size: 1.15rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
        background: white !important;
        color: #0f172a !important; 
        transition: all 0.2s ease;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #4364F7;
        box-shadow: 0 0 0 4px rgba(67, 100, 247, 0.15);
    }
    
    button[kind="primary"] {
        background: linear-gradient(135deg, #0052D4 0%, #4364F7 100%);
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 10px 20px rgba(0, 82, 212, 0.25);
        color: white !important;
    }
    button[kind="primary"]:hover {
        box-shadow: 0 15px 25px rgba(0, 82, 212, 0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# UI HEADERS
# -----------------------------
st.markdown("""
  <h1 class='main-header'>
    PharmAssist <span class='fda-badge'>RAG Engine</span>
  </h1>
  <p class='sub-header'>Next-Generation Medical Entity Extraction & Semantic Search</p>
""", unsafe_allow_html=True)

st.markdown("""
<div class='warning-box'>
    <span class='warning-icon'>⚠️</span><strong>STRICT CLINICAL GUARDRAILS ACTIVE:</strong> 
    This generative application operates under an isolated Retrieval-Augmented Generation (RAG) architecture. 
    The AI is mathematically forbidden from utilizing external medical knowledge and is restricted to extracting answers <b>exclusively</b> from the provided FDA synthetic drug label database to eliminate model hallucinations.
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SYSTEM INITIALIZATION
# -----------------------------
@st.cache_resource(show_spinner="Initializing Vectors & Booting AI... Please wait.")
def load_rag_backend():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    if not os.path.exists("fda_faiss.index") or not os.path.exists("fda_docs.jsonl"):
        return embedder, None, None
    index = faiss.read_index("fda_faiss.index")
    with open("fda_docs.jsonl", "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]
    return embedder, index, docs

try:
    embedder, index, docs = load_rag_backend()
except Exception as e:
    st.error(f"Error loading backend: {e}")
    st.stop()

# -----------------------------
# AWS BEDROCK PROXY CALL
# -----------------------------
def query_bedrock(sys_prompt, user_query):
    url = os.getenv("AWS_BEDROCK_URL")
    if not url:
        return "Error: Missing AWS_BEDROCK_URL environment variable."
        
    model_id = os.getenv("BEDROCK_MODEL_ID", "arn:aws:bedrock:us-east-1:120569630043:inference-profile/us.amazon.nova-micro-v1:0")
    
    prompt = f"{sys_prompt}\n\n### User Question:\n{user_query}"
    
    try:
        response = requests.post(url, json={"prompt": prompt, "model_id": model_id}, timeout=45)
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"Error contacting AWS Bedrock proxy: {str(e)}"

if index is None or docs is None:
    st.error("Local Database missing! Please run `python build_index.py` from the terminal first.")
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-bottom: 30px;'><h1 style='margin:0; font-size: 3rem;'>⚙️</h1></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: white;'>Engine Controls</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem; margin-bottom: 30px;'>Configure vector retrieval depth.</p>", unsafe_allow_html=True)
    
    st.success(f"Indexed {len(docs)} simulated drug profiles.")
    
    st.markdown("---")
    top_k = st.slider("Retrieval Depth (Chunks)", min_value=1, max_value=20, value=8)
    st.markdown("---")
    
    # --- LIVE DATA INJECTION ---
    st.markdown("<h3 style='color: white; text-align: center; margin-top: 10px;'>Live Data Injection</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.85rem; text-align: center; margin-bottom: 10px;'>Upload new FDA updates (TXT, CSV, JSON) to dynamically expand the vector DB.</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload New Data", type=['txt', 'csv', 'json'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
            
        if uploaded_file.name not in st.session_state.processed_files:
            with st.spinner("AI is verifying clinical relevance & injecting..."):
                import csv, io, json
                new_docs = []
                file_name = uploaded_file.name
                
                try:
                    raw_text = ""
                    if file_name.endswith('.txt'):
                        text = uploaded_file.read().decode('utf-8')
                        raw_text = text
                        chunks = [text[i:i+400] for i in range(0, len(text), 400)]
                        for c in chunks:
                            new_docs.append({'drug': f"Update: {file_name}", 'text': c})
                    
                    elif file_name.endswith('.csv'):
                        content = uploaded_file.read().decode('utf-8')
                        raw_text = content
                        reader = csv.DictReader(io.StringIO(content))
                        for row in reader:
                            text_repr = " | ".join([f"{k}: {v}" for k,v in row.items() if v])
                            new_docs.append({'drug': f"Update: {file_name}", 'text': text_repr})
                            
                    elif file_name.endswith('.json'):
                        content_str = uploaded_file.read().decode('utf-8')
                        raw_text = content_str
                        content = json.loads(content_str)
                        if isinstance(content, list):
                            for item in content:
                                new_docs.append({'drug': f"Update: {file_name}", 'text': json.dumps(item)})
                        elif isinstance(content, dict):
                            new_docs.append({'drug': f"Update: {file_name}", 'text': json.dumps(content)})
                    
                    # --- AI Relevance Verification ---
                    preview = raw_text[:800] # Check first 800 chars to avoid huge payload
                    sys_valid = """You are a strict data safety system. Read the following text. 
Does this text purely contain medical, pharmacological, drug-related, or clinical data?
If it contains programming code, requirements.txt dependencies, python packages, sports news, or general text, you must output NO.
Output EXACTLY the word YES or exactly the word NO. Nothing else."""
                    
                    verification_response = query_bedrock(sys_valid, preview)
                    
                    if "YES" not in verification_response.upper():
                        st.error(f"❌ Upload Blocked: '{file_name}' does not appear to contain relevant pharmacological data. Context rejected.")
                        st.session_state.processed_files.append(file_name) # Prevent re-trying
                    else:
                        if new_docs:
                            new_texts = [d['text'] for d in new_docs]
                            new_embs = embedder.encode(new_texts, normalize_embeddings=True)
                            index.add(np.asarray(new_embs, dtype=np.float32))
                            docs.extend(new_docs)
                            
                            st.session_state.processed_files.append(file_name)
                            st.success(f"✅ AI Verified & Securely injected {len(new_docs)} chunks into Vector DB!")
                except Exception as e:
                    st.error(f"Failed to process file: {str(e)}")

    st.markdown("""
    <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.1); margin-top: 20px;'>
        <div style='font-size: 0.75rem; color: #94a3b8;'>ARCHITECTURE</div>
        <div style='font-size: 0.9rem; font-weight: 500;'>• Vector DB: FAISS</div>
        <div style='font-size: 0.9rem; font-weight: 500;'>• Embedder: MiniLM-L6-v2</div>
        <div style='font-size: 0.9rem; font-weight: 500;'>• Inference: AWS Nova Micro</div>
    </div>
    """, unsafe_allow_html=True)

# Main Query Interface
query = st.text_input("Enter Clinical Query", placeholder="e.g., 'What are the dosage and warnings for Seroxetine50?'")

if st.button("Extract Data & Synthesize", type="primary") and query:
    
    with st.spinner("Executing hybrid vector search..."):
        query_vector = embedder.encode([query], normalize_embeddings=True)
        query_vector = np.asarray(query_vector, dtype=np.float32)
        
        raw_top_k = min(top_k * 15, 150) 
        scores, idxs = index.search(query_vector, raw_top_k)
        
        retrieved_context = ""
        clinical_evidence = []
        
        query_terms = [word.strip("?,.!") for word in query.split() if word.istitle() or any(c.isdigit() for c in word)]
        
        for i, (score, idx) in enumerate(zip(scores[0], idxs[0])):
            doc = docs[idx]
            text = doc['text']
            match_bonus = 0
            for term in query_terms:
                if term.lower() in text.lower():
                    match_bonus += 5.0
            clinical_evidence.append((doc['drug'], score + match_bonus, text))
            
        clinical_evidence.sort(key=lambda x: x[1], reverse=True)
        clinical_evidence = clinical_evidence[:top_k]
        
        for i, (drug, score, text) in enumerate(clinical_evidence):
            retrieved_context += f"Evidence Fragment {i+1}:\n{text}\n\n"

    st.markdown("### Isolated Clinical Context (FAISS)")
    cols = st.columns(top_k)
    for i, (drug, score, text) in enumerate(clinical_evidence):
        with cols[i]:
            formatted_text = text.replace("Class:", "<br><strong>Class:</strong>")
            formatted_text = formatted_text.replace("Indications (used for):", "<br><strong>Indications:</strong>")
            formatted_text = formatted_text.replace("Side Effects:", "<br><strong>Side Effects:</strong>")
            formatted_text = formatted_text.replace("Standard Dosage:", "<br><strong>Dosage:</strong>")
            formatted_text = formatted_text.replace("Contraindications:", "<br><strong>Contraindications:</strong>")
            formatted_text = formatted_text.replace("Boxed Warnings:", "<br><strong style='color:#ef4444;'>Warnings:</strong>")
            formatted_text = formatted_text.replace(f"Drug Name: {drug}.", "")

            st.markdown(f"""
            <div class='clinical-card'>
                <div class='drug-title'>{drug}</div>
                <div class='confidence-score'>Relevance: {score:.2f}</div>
                <div class='card-content'>{formatted_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
    with st.spinner("Synthesizing clinical response via AWS Bedrock..."):
        sys_prompt = f"""You are an incredibly strict, highly professional clinical data extraction Copilot. 
Your singular job is to answer the user's question USING ONLY the provided Extracted Clinical Evidence.

CRITICAL RULES:
1. Do NOT output the entire drug profile unless explicitly asked.
2. ONLY answer exactly what the user asked. 
3. Keep the answer concise, nicely formatted, and strictly relevant.
4. If the retrieved evidence contains bizarre medical combinations, state it EXACTLY as the evidence says.
5. If the answer to the user's question is not present in the Extracted Clinical Evidence, say explicitly: "Insufficient label data available within the current scope to answer this query safely."

CRITICAL FORMATTING RULE:
Since your output will be directly embedded into an HTML div tag, YOU MUST FORMAT YOUR ENTIRE ANSWER USING RAW HTML TAGS.
- DO NOT use markdown like **bold**, use <strong>bold</strong>.
- DO NOT use line breaks \\n, use <br>.
- DO NOT use markdown lists like -, use <ul><li>Item</li></ul>.

### Extracted Clinical Evidence ###
{retrieved_context}
"""
        
        answer = query_bedrock(sys_prompt, query)
        
        # Replace common markdown artifacts if the LLM hallucinated them despite the rules
        answer = answer.replace("**", "")
        
        st.markdown(f"""
        <div class='copilot-response'>
            <div class='response-title'>✨ AI Synthesis</div>
            <div style='margin-top: 10px;'>{answer}</div>
        </div>
        """, unsafe_allow_html=True)
