"""Streamlit UI for the Upwork API Support Bot.

Run from the project root: streamlit run app.py
"""
from __future__ import annotations

import time

import streamlit as st

from src import llm, rag
from src.config import (
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
)

st.set_page_config(
    page_title="Upwork API Support Bot",
    layout="wide",
)

@st.cache_resource(show_spinner="Loading vector store...")
def get_vectorstore():
    return rag.load_vectorstore()


EXAMPLE_QUESTIONS = [
    "What is the rate limit for the Upwork API?",
    "How long is an OAuth access token valid for?",
    "Can I use a Client Credentials Grant to access a user's private contract details?",
    "Who can create subscriptions on the Upwork API?",
]


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def render_sources(chunks) -> None:
    with st.expander(f"Retrieved Context ({len(chunks)} chunks)"):

        for i, chunk in enumerate(chunks, 1):
            page = chunk.metadata.get("page", "?")

            st.markdown(f"**Chunk {i}**")
            st.caption(f"Page {page}")

            st.code(
                chunk.page_content,
                language="text",
            )


def answer_question(question: str, vectorstore) -> None:
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("assistant"):

        with st.status("Processing request", expanded=True) as status:

            st.write("Embedding query")

            try:
                chunks = rag.retrieve(question, vectorstore)

            except Exception as e:
                st.error(f"Retrieval error: {e}")
                return

            st.write(f"Retrieved candidate documents")
            st.write(f"Selected top {len(chunks)} chunks")
            st.write("Generating response")

            status.update(
                label="Completed",
                state="complete",
            )

        placeholder = st.empty()

        buffer = ""

        elapsed = 0.0

        try:
            start = time.perf_counter()

            for delta in llm.stream_answer(question, chunks):
                buffer += delta
                placeholder.markdown(buffer + "▌")

            elapsed = time.perf_counter() - start

            placeholder.markdown(buffer)

        except Exception as e:
            placeholder.error(f"LLM error: {e}")
            return

        st.caption(
            f"Response time: {elapsed:.2f}s • "
            f"Retrieved: top-{TOP_K} context"
        )

        render_sources(chunks)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": buffer,
            "latency": elapsed,
            "chunks": chunks,
        }
    )


init_state()

with st.sidebar:

    st.header("Upwork API Bot")

    st.subheader("Configuration")

    st.markdown(
        f"""
**Embedding Model**

{EMBEDDING_MODEL}

**LLM**

{LLM_MODEL}

**Chunk Size**

{CHUNK_SIZE}

**Top K**

{TOP_K}
"""
    )

    st.subheader("Example Questions")

    for i, q in enumerate(EXAMPLE_QUESTIONS):

        if st.button(q, key=f"ex_{i}"):

            st.session_state.pending_question = q

            st.rerun()

    st.divider()

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()


st.title("Upwork API Support Bot")

st.caption(
    "Answers are generated only from retrieved Upwork API documentation. "
    "If the documentation does not support an answer, the assistant explicitly states that the information is unavailable."
)

try:
    vectorstore = get_vectorstore()

except Exception as e:
    st.error(str(e))
    st.stop()


for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and "chunks" in msg:

            st.caption(
                f"Response time: {msg['latency']:.2f}s • "
                f"Retrieved: top-{len(msg['chunks'])} context"
            )

            render_sources(msg["chunks"])


question = st.chat_input(
    "Ask about OAuth, endpoints, authentication, rate limits..."
)

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    answer_question(question, vectorstore)
