import streamlit as st
from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.vector_store import create_vector_store
from utils.qa_chain import create_qa_chain

st.set_page_config(page_title="Document Q&A")

st.title("📄 Document Q&A System")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        text = load_pdf(uploaded_file)
        chunks = split_text(text)
        vectorstore = create_vector_store(chunks)
        qa_chain = create_qa_chain(vectorstore)

    st.success("Document processed!")

    query = st.text_input("Ask a question from the document:")

    if query:
        answer = qa_chain(query)
        st.write("### Answer")
        st.write(answer)
