import streamlit as st
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

st.set_page_config(page_title="Indian Maritime Law Bot", page_icon="⚓", layout="centered")

# Custom CSS injection to fix padding issues on mobile Chrome
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    @media (max-width: 640px) {
        .stChatMessage { padding: 0.5rem; }
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚓ Indian Merchant Shipping Act Bot")
st.caption("Instant Mobile Q&A — Powered by Groq LPU & RAG")

# API Key Check (Prioritize Streamlit Secrets, fallback to manual entry)
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
else:
    sidebar_placeholder = st.sidebar.empty()
    api_key = sidebar_placeholder.text_input("Enter Groq API Key to test:", type="password")
    st.sidebar.markdown("[Get a free key here](https://console.groq.com)")

# Initialize Vector DB
@st.cache_resource
def initialize_vector_db():
    if os.path.exists("merchant_shipping_act.txt"):
        with open("merchant_shipping_act.txt", "r", encoding="utf-8") as f:
            text_data = f.read()
    else:
        text_data = "Merchant Shipping Act 2025. Section 15: Rules for vessel registration under the Indian Flag. Section 59: Minimum age for seafarers is 16."
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = text_splitter.create_documents([text_data])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings)

if not api_key:
    st.info("← Please add your Groq API key in the sidebar (or configure it in Streamlit Secrets) to start chatting.")
else:
    # CLEANUP STEP: If the key is validated, collapse the sidebar completely for mobile users
    if "GROQ_API_KEY" in st.secrets:
        st.markdown("<style>section[data-testid='stSidebar'] {display: none;}</style>", unsafe_allow_html=True)

    try:
        client = Groq(api_key=api_key)
        db = initialize_vector_db()
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Ask a legal question..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            relevant_chunks = db.similarity_search(user_query, k=2)
            context = "\n\n".join([doc.page_content for doc in relevant_chunks])

            system_prompt = (
                "You are an expert maritime legal assistant specializing in the Indian Merchant Shipping Act 2025. "
                "Answer using ONLY the provided text context. If unsure, state that it isn't in the current text. Cite sections explicitly.\n\n"
                f"--- CONTEXT ---\n{context}"
            )

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                completion = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    stream=True,
                )
