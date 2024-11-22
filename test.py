import os
from docx import Document
import fitz  # PyMuPDF
from PIL import Image
import io

# Paths for input and output folders
input_folder = "uploaded_documents"
output_folder = "uploaded_images"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to save an image to the output folder
def save_image(image, filename):
    output_path = os.path.join(output_folder, filename)
    image.save(output_path)
    print(f"Saved image: {output_path}")

# Function to extract images from .docx files
def extract_images_from_docx(doc_path):
    doc = Document(doc_path)
    for i, rel in enumerate(doc.part.rels.values()):
        if "image" in rel.target_ref:
            image_data = rel.target_part.blob
            image = Image.open(io.BytesIO(image_data))
            save_image(image, f"{os.path.basename(doc_path)}_image_{i + 1}.png")

# Function to extract images from .pdf files
def extract_images_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images(full=True)
        for i, img in enumerate(images):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_data = base_image["image"]
            image = Image.open(io.BytesIO(image_data))
            save_image(image, f"{os.path.basename(pdf_path)}_page_{page_num + 1}_image_{i + 1}.png")

# Loop through all files in the input folder
for filename in os.listdir(input_folder):
    file_path = os.path.join(input_folder, filename)
    if filename.lower().endswith(".docx"):
        extract_images_from_docx(file_path)
    elif filename.lower().endswith(".pdf"):
        extract_images_from_pdf(file_path)

print("Image extraction completed.")
