from rag import read_pdf, create_chunks, create_faiss_index_from_chunks, process_multi_question
import json

print("1. Reading PDF sample.pdf...")
text = read_pdf("uploads/sample.pdf")
print(f"Extracted {len(text)} characters.")

print("2. Chunking text...")
chunks = create_chunks(text)
print(f"Created {len(chunks)} chunks.")

print("3. Building FAISS index...")
index, model = create_faiss_index_from_chunks(chunks)
print(f"FAISS index built with {index.ntotal} vectors.")

user_prompt = "Hello! Can you tell me what 5G is? Also, what are its main features and how fast can it transmit data?"
print(f"\n4. Processing multi-question user input:\n\"{user_prompt}\"")

res = process_multi_question(user_prompt, chunks=chunks, index=index)
print("\n--- OUTPUT RESULT ---")
print(json.dumps(res, indent=2))
