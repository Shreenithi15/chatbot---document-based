from pypdf import PdfReader

reader = PdfReader("uploads/sample.pdf")

print("PDF opened successfully!")

print("Pages:", len(reader.pages))

for i, page in enumerate(reader.pages):
    print(f"\nReading page {i+1}...")

    text = page.extract_text()

    if text is None:
        print("No text found.")
    else:
        print("Characters extracted:", len(text))
        print(text[:500])