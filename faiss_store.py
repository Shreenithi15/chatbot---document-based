from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Read PDF
reader = PdfReader("uploads/sample.pdf")

text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Chunks created: {len(chunks)}")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype="float32"))

print("Vectors stored in FAISS:", index.ntotal)