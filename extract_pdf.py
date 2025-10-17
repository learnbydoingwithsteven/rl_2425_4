import fitz  # PyMuPDF

pdf_path = "reinforcement_learning_projects_2024-25.pdf"
doc = fitz.open(pdf_path)

full_text = ""
for page_num in range(len(doc)):
    page = doc[page_num]
    full_text += f"\n--- Page {page_num + 1} ---\n"
    full_text += page.get_text()

with open("pdf_content.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"Extracted {len(doc)} pages")
print(f"Total characters: {len(full_text)}")
