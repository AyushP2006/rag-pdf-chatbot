from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Read the PDF
pdf_path = "data/sample.pdf"
reader = PdfReader(pdf_path)

text = ""

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""
    text += page_text + "\n"
    # Fixed: Indented to be inside the for loop
    print(f"Extracted page {page_number}")

# Step 2: Check extracted text
if not text.strip():
    # Fixed: Indented to be inside the if statement
    raise ValueError("No readable text found in the PDF.")

print(f"\nTotal characters: {len(text)}")

# Step 3: Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Step 4: Split the text
chunks = text_splitter.split_text(text)

# Step 5: Display the chunks
print(f"\nTotal chunks created: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    # Fixed: Indented to be inside the for loop
    print(f"\n--- Chunk {index} ---")
    print(chunk)
