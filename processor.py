import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from process_folder import extract_text, is_image_file, load_image_pil, get_image_hash
import cv2

def orb_embedding(image_pil):
    try:
        img = np.array(image_pil)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=500)
        keypoints, descriptors = orb.detectAndCompute(img_gray, None)

        if descriptors is None:
            return None

        # Normalize descriptor vector
        desc = descriptors.flatten()
        if len(desc) < 500:
            desc = np.pad(desc, (0, 500 - len(desc)), 'constant')
        else:
            desc = desc[:500]

        desc = desc.astype('float32')
        norm = np.linalg.norm(desc)
        if norm > 0:
            desc = desc / norm

        return desc

    except:
        return None


def cosine(a, b):
    if a is None or b is None:
        return 0
    dp = np.dot(a, b)
    dp = max(-1, min(1, dp))  # numerical stability
    return dp


def process_documents(file_paths):
    text_files = []
    text_contents = []

    image_files = []
    image_hashes = []
    image_embeddings = []

    # Load data
    for file in tqdm(file_paths, desc="Loading files"):
        if is_image_file(file):
            image_files.append(file)

            # Hash
            h = get_image_hash(file)
            image_hashes.append(h)

            # ORB embedding
            pil_img = load_image_pil(file)
            emb = orb_embedding(pil_img)
            image_embeddings.append(emb)

        else:
            txt = extract_text(file)
            text_files.append(file)
            text_contents.append(txt if txt is not None else "")

    # TEXT SIMILARITY
    text_results = []
    if len(text_contents) >= 2:
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(text_contents)
        sim = cosine_similarity(tfidf, tfidf)

        for i in range(len(text_files)):
            for j in range(i+1, len(text_files)):
                score = round(float(sim[i][j]) * 100, 2)
                text_results.append({
                    "file1": text_files[i],
                    "file2": text_files[j],
                    "similarity": score,
                    "type": "text"
                })

    # IMAGE SIMILARITY
    image_results = []
    for i in range(len(image_files)):
        for j in range(i+1, len(image_files)):

            # PERCEPTUAL HASH
            h1 = image_hashes[i]
            h2 = image_hashes[j]

            phash_sim = 0
            if h1 and h2:
                import imagehash
                dist = (imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2))
                phash_sim = round((1 - dist / 64) * 100, 2)

            # ORB SIMILARITY
            orb_sim = round(cosine(image_embeddings[i], image_embeddings[j]) * 100, 2)

            # Combined Score (best results)
            combined = round((phash_sim * 0.4) + (orb_sim * 0.6), 2)

            image_results.append({
                "file1": image_files[i],
                "file2": image_files[j],
                "similarity": combined,
                "type": "image"
            })

    # COMBINE RESULTS
    all_results = text_results + image_results
    df = pd.DataFrame(all_results).sort_values(by="similarity", ascending=False)

    # Save CSV
    csv_out = os.path.join("uploads", "similarity_report.csv")
    df.to_csv(csv_out, index=False)

    # HTML Table
    html = df.to_html(index=False)

    return html, csv_out
