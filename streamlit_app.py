# streamlit_app.py

import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000/api/v1"

st.set_page_config(
    page_title="RAG System",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG System")
st.caption("Upload PDFs and ask questions — answers grounded in your documents.")

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Ingest Documents", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Ingesting documents..."):
                files = [
                    ("files", (f.name, f.getvalue(), "application/pdf"))
                    for f in uploaded_files
                ]
                try:
                    resp = requests.post(f"{API_BASE}/ingest", files=files)
                    data = resp.json()
                    if resp.status_code == 200:
                        st.success(f"✅ Ingested {data['total_files_in_index']} file(s)")
                        st.info(f"📊 {data['total_chunks']} chunks indexed")
                        st.session_state.index_built = True
                    else:
                        st.error(f"Error: {data}")
                except Exception as e:
                    st.error(f"Could not connect to API: {e}")

    st.divider()

    if st.button("🗑️ Reset Index", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/ingest/reset")
            st.session_state.index_built = False
            st.session_state.chat_history = []
            st.success("Index reset.")
        except Exception as e:
            st.error(f"Error: {e}")

    # Health status
    try:
        health = requests.get(f"{API_BASE}/health").json()
        if health.get("index_built"):
            st.success(f"✅ Index active — {health['total_chunks']} chunks")
        else:
            st.warning("⚠️ No documents ingested yet")
    except:
        st.error("❌ API not running")

# ── Chat Interface ────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "index_built" not in st.session_state:
    st.session_state.index_built = False

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['source']} | Page {s['page']} | Score: {s['score']:.3f}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.index_built:
        st.warning("Please ingest documents first using the sidebar.")
    else:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/query",
                        json={"question": prompt, "top_k": 3}
                    )
                    data = resp.json()

                    if resp.status_code == 200:
                        answer = data["answer"]
                        sources = data["sources"]
                        st.write(answer)
                        with st.expander("📎 Sources"):
                            for s in sources:
                                st.caption(f"📄 {s['source']} | Page {s['page']} | Score: {s['score']:.3f}")

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        })
                    else:
                        st.error(f"Error: {data}")

                except Exception as e:
                    st.error(f"Could not connect to API: {e}")