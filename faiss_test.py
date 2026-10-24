
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


# Step 1: Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError(
        "GOOGLE_API_KEY not found. Check your .env file."
    )


# Step 2: Read the PDF
pdf_path = "data/sample.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(
        f"PDF file not found: {pdf_path}"
    )

reader = PdfReader(pdf_path)
pages = []

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""

    if page_text.strip():
        pages.append(page_text)

if not pages:
    raise ValueError(
        "No readable text found in the PDF."
        " The PDF may be scanned and require OCR."
    )

print(
    f"PDF loaded successfully: "
    f"{len(pages)} pages contain text."
)


# Step 3: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.create_documents(pages)

if not chunks:
    raise ValueError("No text chunks were created.")

print(f"Total chunks created: {len(chunks)}")


# Step 4: Initialize the embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
)


# Step 5: Create the FAISS vector store
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings_model
)

print("Embeddings stored in FAISS successfully!")


# Step 6: Save the vector store locally
vector_store.save_local("vectorstore")

print("Vector store saved in the 'vectorstore' folder.")


# Step 7: Search the PDF
while True:
    question = input(
        "\nAsk a question about your PDF "
        "(type 'exit' to quit): "
    ).strip()

    if question.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    results = vector_store.similarity_search(
        question,
        k=3
    )

    print("\n--- Relevant passages ---")

    if not results:
        print("No relevant passages found.")
        continue

    for index, document in enumerate(results, start=1):
        print(f"\nPassage {index}:")
        print(document.page_content)

