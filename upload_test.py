
import streamlit as st

st.set_page_config(page_title="PDF Upload Test")

st.title("Multiple PDF Upload Test")

uploaded_files = st.file_uploader(
    "Select multiple PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    key="test_pdf_uploader",
)

if uploaded_files:
    st.success(f"Total PDFs selected: {len(uploaded_files)}")

    for pdf in uploaded_files:
        st.write(f"📄 {pdf.name}")

