"""
Assistant RH - Prototype RAG (Retrieval-Augmented Generation)
Projet de démonstration — Entretien Michelin IA & HR
Auteur : Imad El Khelyfy
"""

import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- Config ---
st.set_page_config(
    page_title="Assistant RH — Prototype RAG",
    page_icon="🤖",
    layout="centered"
)

# --- Style ---
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTextInput > div > div > input { border-radius: 8px; }
    .answer-box {
        background: white;
        color: #111827;
        border-left: 4px solid #1a56db;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-top: 1rem;
        font-size: 15px;
        line-height: 1.7;
    }
    .source-box {
        background: #f0f4ff;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 13px;
        color: #444;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🤖 Assistant RH Michelin")
st.caption("Prototype RAG — Posez vos questions sur les politiques RH internes")
st.divider()

# --- API Key ---
api_key_default = st.secrets.get("GEMINI_API_KEY", "")
api_key = st.sidebar.text_input(
    "🔑 Gemini API Key",
    value=api_key_default,
    type="password",
    help="Obtenez votre clé sur https://aistudio.google.com"
)

if not api_key:
    st.info("👈 Entrez votre clé Gemini API dans la barre latérale pour commencer.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# --- Load and index document ---
@st.cache_resource(show_spinner="📚 Indexation des documents RH...")
def build_index():
    loader = TextLoader("hr_policy.txt", encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

try:
    vectorstore = build_index()
except Exception as e:
    st.error(f"Erreur lors de l'indexation : {e}")
    st.stop()

# --- RAG Chain ---
@st.cache_resource(show_spinner=False)
def build_chain(_vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.1,  # faible température = réponses plus fiables
    )

    prompt_template = """Tu es un assistant RH interne. Réponds uniquement à partir des documents fournis.
Si la réponse n'est pas dans les documents, dis clairement que tu ne sais pas — ne invente jamais.

Contexte :
{context}

Question : {question}

Réponse :"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=_vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

chain = build_chain(vectorstore)

# --- Question examples ---
st.markdown("**Questions exemples :**")
cols = st.columns(3)
example_questions = [
    "Combien de jours de congés par an ?",
    "Comment fonctionne le télétravail ?",
    "Quelle est la politique de formation ?"
]
for i, q in enumerate(example_questions):
    if cols[i].button(q, use_container_width=True):
        st.session_state["question"] = q

# --- Input ---
question = st.text_input(
    "Votre question :",
    value=st.session_state.get("question", ""),
    placeholder="Ex: Quels sont mes droits en cas d'arrêt maladie ?",
    key="question"
)

if st.button("Poser la question ↗", type="primary") and question:
    with st.spinner("Recherche dans les documents RH..."):
        try:
            result = chain.invoke({"query": question})
            answer = result["result"]
            sources = result["source_documents"]

            st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

            with st.expander("📄 Sources utilisées"):
                for i, doc in enumerate(sources):
                    st.markdown(
                        f'<div class="source-box"><b>Extrait {i+1} :</b><br>{doc.page_content}</div>',
                        unsafe_allow_html=True
                    )
        except Exception as e:
            st.error(f"Erreur : {e}")

# --- Footer ---
st.divider()
st.caption("Prototype RAG · Python · LangChain · Gemini AI · FAISS · Imad El Khelyfy")
