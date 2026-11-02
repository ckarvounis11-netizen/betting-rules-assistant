  ## About this project

  This is a learning project. I gathered and translated the rule documents
  and built the app with the help of AI coding tools, to understand how
  retrieval-augmented generation (RAG) works in practice.

## What it does

- Answers questions about the sportsbook rules of six operators: DLS (Danske
  Licens Spil), CATSR (Casino Arizona & Talking Stick Resort), BCLC (PlayNow),
  LQ (Loto-Quebec, Mise-o-jeu+), TOTO and WSB.
- The source documents were merged into one PDF (about 300 pages). Dutch and
  Danish parts were translated into English.
- Answers questions on voiding, settlement and grading for one operator or
  across all of them, replies in the language of the question, and shows the
  excerpts it used.

## How it works (RAG)

1. The PDF is split into chunks, each tagged with its operator and page.
2. Chunks are turned into embeddings (OpenAI) and stored in a FAISS index.
3. For each question, the most relevant chunks are retrieved.
4. A chat model answers using only those excerpts and cites operator and page.

## Tech stack

Python, Streamlit, LangChain, OpenAI API, FAISS, pdfplumber.

## Limitations

- The TOTO (Dutch) and DLS (Danish) sections are condensed translations, not
  word for word. Check the original documents when exact wording matters.
- Answers are generated from retrieved excerpts and can be wrong or
  incomplete. Always verify important decisions against the source rules.
- Personal project, not affiliated with any of the operators listed.
