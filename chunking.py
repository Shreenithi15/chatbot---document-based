from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Read the PDF
reader = PdfReader("uploads/sample.pdf")

text = ""

for page in reader.pages:
    extracted = page.extract_text()
    if extracted:
        text += extracted

print("Total characters:", len(text))

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))

print("\nFirst Chunk:\n")
print(chunks[0])