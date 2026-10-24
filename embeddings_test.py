import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API key from .env
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

# Initialize the embedding model
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key,
)

# Sample sentences
sentences = [
    "Python is a programming language.",
    "Python is useful for software development.",
    "Cricket is a popular sport.",
]

# Generate embeddings
vectors = embeddings_model.embed_documents(sentences)

# Display results
for sentence, vector in zip(sentences, vectors):
    print("\nSentence:", sentence)
    print("Vector dimensions:", len(vector))
    print("First 5 numbers:", vector[:5])