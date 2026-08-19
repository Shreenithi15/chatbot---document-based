from pypdf import PdfReader

reader = PdfReader("uploads/offer.pdf")

print(f"Number of pages: {len(reader.pages)}")

for i, page in enumerate(reader.pages):
    text = page.extract_text()

    print(f"\n----- Page {i+1} -----")

    if text:
        print(text[:500])   # Print the first 500 characters
    else:
        print("No text found on this page.")