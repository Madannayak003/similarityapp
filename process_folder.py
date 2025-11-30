from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import imagehash
import numpy as np
import os

def extract_text(path):
    p = path.lower()

    if p.endswith(".txt"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except:
            return ""

    if p.endswith(".pdf"):
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except:
            return ""

    if p.endswith(".docx"):
        try:
            doc = Document(path)
            return "\n".join([p.text for p in doc.paragraphs])
        except:
            return ""

    return ""


def is_image_file(path):
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"]


def load_image_pil(path):
    try:
        img = Image.open(path).convert("RGB")
        return img
    except:
        return None


def get_image_hash(path):
    try:
        img = Image.open(path).convert("L")
        return str(imagehash.phash(img))
    except:
        return None
