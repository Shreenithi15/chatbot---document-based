from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Step 1: Read the PDF
reader = PdfReader("uploads/sample.pdf")

text = ""

for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")

# Step 3: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 4: Convert chunks into embeddings
embeddings = model.encode(chunks)

print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst embedding (first 10 values):")
print(embeddings[0][:10])