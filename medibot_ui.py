import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

DB_FAISS_PATH = "vectorstore/db_faiss"

st.set_page_config(page_title="Medical Bot · Medical Reference Assistant", page_icon="🩺", layout="centered")

# ---------- Styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --teal: #0F6E6E;
    --mint: #E8F6F3;
    --coral: #D97757;
    --ink: #000000;
    --paper: #FAFAF8;
    --line: #D7E4E2;
}

html, body, [class*="css"], p, span, div, h1, h2, h3 {
    font-family: 'Inter', sans-serif;
    color: #000000 !important;
}
.stApp { background: var(--paper); }

.medibot-header h1 { font-family: 'Lora', serif; font-weight: 600; font-size: 2.1rem; margin-bottom: 0.1rem; }
.medibot-header p { font-size: 0.95rem; margin-top: 0; }

.pulse-divider { width: 100%; height: 28px; margin: 0.4rem 0 1.2rem 0; }
.pulse-divider path {
    stroke: var(--teal); stroke-width: 2; fill: none;
    stroke-dasharray: 300; stroke-dashoffset: 300;
    animation: draw 1.8s ease-out forwards;
}
@keyframes draw { to { stroke-dashoffset: 0; } }

.disclaimer-banner {
    background: #FBEEE8; border-left: 3px solid var(--coral);
    padding: 0.6rem 0.9rem; border-radius: 6px; font-size: 0.85rem; margin-bottom: 1.2rem;
}

.msg-row { display: flex; margin-bottom: 0.9rem; gap: 0.6rem; align-items: flex-start; }
.msg-row.user { justify-content: flex-end; }
.msg-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.95rem; flex-shrink: 0;
}
.msg-avatar.bot { background: var(--teal); }
.msg-avatar.user { background: #DDE6E5; }

.msg-bubble { padding: 0.65rem 0.9rem; border-radius: 10px; max-width: 78%; line-height: 1.5; font-size: 0.95rem; }
.msg-bubble.bot { background: white; border: 1px solid var(--line); border-left: 3px solid var(--teal); }
.msg-bubble.user { background: var(--mint); }

.source-card {
    background: white; border: 1px solid var(--line); border-radius: 6px;
    padding: 0.5rem 0.7rem; margin-bottom: 0.5rem;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
}
.source-card .src-label { font-weight: 500; }

section[data-testid="stSidebar"] { background: #F1F8F6; border-right: 1px solid var(--line); }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🩺 MediBot")
    st.markdown("An assistant that answers questions using *The Gale Encyclopedia of Medicine* as its only source of truth.")
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("1. Your question is matched against the encyclopedia\n2. Relevant passages are retrieved\n3. The model answers using only those passages")
    st.markdown("---")
    st.caption("Source: The Gale Encyclopedia of Medicine, 2nd Edition")

# ---------- Header (single-line HTML, no leading indentation) ----------
header_html = (
    '<div class="medibot-header"><h1>Medical Bot</h1>'
    '<p>Grounded answers from a trusted medical encyclopedia</p></div>'
    '<svg class="pulse-divider" viewBox="0 0 400 28" preserveAspectRatio="none">'
    '<path d="M0,14 L130,14 L145,2 L160,26 L175,14 L400,14" /></svg>'
)
st.markdown(header_html, unsafe_allow_html=True)

# st.markdown(
#     '<div class="disclaimer-banner">⚠️ Educational use only — not a substitute '
#     'for professional medical advice. Always consult a doctor for diagnosis or treatment.</div>',
#     unsafe_allow_html=True,
# )

# ---------- Helpers ----------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])


def render_message(role, content):
    is_user = role == "user"
    avatar = "🙂" if is_user else "🩺"
    cls = "user" if is_user else "bot"
    avatar_html = f'<div class="msg-avatar {cls}">{avatar}</div>'
    bubble_html = f'<div class="msg-bubble {cls}">{content}</div>'
    inner = bubble_html + avatar_html if is_user else avatar_html + bubble_html
    row_class = "msg-row user" if is_user else "msg-row"
    st.markdown(f'<div class="{row_class}">{inner}</div>', unsafe_allow_html=True)


def render_sources(source_documents):
    with st.expander(f"📚 Reference passages ({len(source_documents)})"):
        for doc in source_documents:
            page = doc.metadata.get("page_label", "?")
            source = os.path.basename(doc.metadata.get("source", "encyclopedia"))
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:220].rsplit(" ", 1)[0] + "…"
            card_html = f'<div class="source-card"><span class="src-label">{source} · p.{page}</span><br>{snippet}</div>'
            st.markdown(card_html, unsafe_allow_html=True)


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know — don't try to make up an answer.
Don't provide anything outside the given context.
This is for educational reference only, not medical advice.

Context: {context}
Question: {question}

Start the answer directly. No small talk.
"""

# ---------- Chat state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    render_message(message["role"], message["content"])
    if message["role"] == "assistant" and message.get("sources"):
        render_sources(message["sources"])

prompt = st.chat_input("Ask about a condition, treatment, or term…")

if prompt:
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        with st.spinner("Searching the encyclopedia…"):
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="openai/gpt-oss-120b",
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)},
            )
            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            sources = response["source_documents"]

        render_message("assistant", result)
        render_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": result, "sources": sources})

    except Exception as e:
        st.error(f"Error: {str(e)}")