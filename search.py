from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ----------------------------
# Step 1: Read PDF
# ----------------------------
reader = PdfReader("uploads/sample.pdf")

text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted

# ----------------------------
# Step 2: Split into chunks
# ----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# ----------------------------
# Step 3: Create embeddings
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# ----------------------------
# Step 4: Create FAISS index
# ----------------------------
index = faiss.IndexFlatL2(embeddings.shape[1])

index.add(np.array(embeddings, dtype="float32"))

# ----------------------------
# Step 5: User Question
# ----------------------------
question = "What is 5G?"

question_embedding = model.encode([question])

# ----------------------------
# Step 6: Search
# ----------------------------
distance, index_result = index.search(
    np.array(question_embedding, dtype="float32"),
    k=1
)

best_chunk = chunks[index_result[0][0]]

print("\nQuestion:")
print(question)

print("\nMost Relevant Chunk:\n")
print(best_chunk)