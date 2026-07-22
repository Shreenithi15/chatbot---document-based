from pypdf import PdfReader

reader = PdfReader("C:\Users\Shreenithi\Downloads\5G technology report (1).pdf")

text = ""

for page in reader.pages:
    text += page.extract_text()

print(text)