from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pdfplumber
import streamlit as st
import os

# =========================================================
# 1. SETTINGS
# =========================================================

MY_OPENAI_KEY = st.secrets["OPENAI_API_KEY"]

PDF_FILE = "combined_rules.pdf"   # must be in the same folder as this file

# First PDF page of each company's section
COMPANIES = [
    ("DLS", 1),
    ("CATSR", 21),
    ("BCLC", 49),
    ("LQ", 129),
    ("TOTO", 159),
    ("WSB", 189),
]

COMPANY_NAMES = {
    "DLS": "Danske Licens Spil",
    "CATSR": "Casino Arizona & Talking Stick Resort",
    "BCLC": "BCLC / PlayNow",
    "LQ": "Loto-Quebec / Mise-o-jeu+",
    "TOTO": "TOTO Online B.V.",
    "WSB": "World Sports Betting",
}

st.set_page_config(page_title="Betting Rules Assistant", page_icon="📚")
st.header("Betting Rules Assistant")


def company_of(page_number):
    name = COMPANIES[0][0]
    for code, start in COMPANIES:
        if page_number >= start:
            name = code
    return name


# =========================================================
# 2. LOAD PDF -> one Document per chunk, with company + page
# =========================================================

@st.cache_resource(show_spinner="Reading the rules document...")
def load_chunks():
    if not os.path.exists(PDF_FILE):
        return None, f"File '{PDF_FILE}' was not found in the repository."

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1500,
        chunk_overlap=250,
    )

    chunks = []
    with pdfplumber.open(PDF_FILE) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").replace("\t", " ").strip()
            if not text:
                continue
            company = company_of(page_number)
            for piece in splitter.split_text(text):
                chunks.append(
                    Document(
                        # company name is inside the text too, so searches
                        # like "BCLC tennis retirement" match better
                        page_content=f"[{company}] {piece}",
                        metadata={"company": company, "page": page_number},
                    )
                )
    return chunks, None


chunks, error_message = load_chunks()
if error_message:
    st.error(error_message)
    st.stop()


# =========================================================
# 3. EMBEDDINGS + FAISS
# =========================================================

@st.cache_resource(show_spinner="Building the search index (first run only)...")
def create_vector_store():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=MY_OPENAI_KEY,
    )
    return FAISS.from_documents(chunks, embeddings)


vector_store = create_vector_store()


@st.cache_resource
def create_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=1200,
        openai_api_key=MY_OPENAI_KEY,
    )


llm = create_llm()


# =========================================================
# 4. PROMPT
# =========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an internal Betting Rules Assistant. Answer strictly from "
        "the rule excerpts below. Each excerpt is labelled with its company "
        "code and PDF page.\n\n"
        "Company codes: DLS = Danske Licens Spil; CATSR = Casino Arizona & "
        "Talking Stick Resort; BCLC = BCLC/PlayNow; LQ = Loto-Quebec "
        "(Mise-o-jeu+); TOTO = TOTO Online B.V.; WSB = World Sports Betting.\n\n"
        "RULES:\n"
        "1. Use ONLY the excerpts. No outside knowledge, no guessing.\n"
        "2. Rules differ between companies. Always say which company's rule "
        "you are quoting. If the question names no company and the "
        "excerpts show different rules for different companies, list each "
        "company's rule separately.\n"
        "3. If the excerpts do not contain the answer, say clearly that "
        "there is not enough information in the available rules. Do not "
        "fill the gap.\n"
        "4. For voiding / settlement / grading questions, explain the rule "
        "that supports the answer.\n"
        "5. Finish with the sources, in the form: Company - page N.\n"
        "6. Reply in the same language as the question.\n\n"
        "EXCERPTS:\n{context}",
    ),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata['company']} | page {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )


# =========================================================
# 5. UI
# =========================================================

options = ["All companies"] + [code for code, _ in COMPANIES]
choice = st.selectbox(
    "Company",
    options,
    format_func=lambda c: c if c == "All companies" else f"{c} - {COMPANY_NAMES[c]}",
)

user_question = st.text_input("Ask a question about the betting rules:")

if user_question:
    with st.spinner("Searching the betting rules..."):
        if choice == "All companies":
            # one lookup per company, so every company is represented
            docs = []
            for code, _ in COMPANIES:
                docs += vector_store.max_marginal_relevance_search(
                    user_question, k=3, fetch_k=40, filter={"company": code}
                )
        else:
            docs = vector_store.max_marginal_relevance_search(
                user_question, k=8, fetch_k=60, filter={"company": choice}
            )

        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": format_docs(docs), "question": user_question}
        )

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Excerpts used"):
        for d in docs:
            st.markdown(f"**{d.metadata['company']} - page {d.metadata['page']}**")
            st.caption(d.page_content[:400] + "...")
