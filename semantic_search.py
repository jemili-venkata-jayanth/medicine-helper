"""
semantic_search.py
-------------------
Builds a semantic search index over the medicine dataset so that queries
like "fever tablet" or "dolo" or "medicine for allergy" can match the
right medicine even without exact word overlap.

Uses sentence-transformers to convert text into embeddings (vectors that
capture meaning), and FAISS to quickly find the closest match.
"""

import json
import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = os.path.join(os.path.dirname(__file__), "medicines.json")

# Small, fast, good-quality embedding model. Downloads once (~80MB) and caches locally.
MODEL_NAME = "all-MiniLM-L6-v2"


class MedicineSearchEngine:
    def __init__(self):
        print("Loading embedding model (first run downloads the model)...")
        self.model = SentenceTransformer(MODEL_NAME)

        with open(DATA_PATH, "r", encoding="utf-8") as f:
            self.medicines = json.load(f)

        self.index, self.searchable_texts, self.text_to_medicine = self._build_index()
        print(f"Index built for {len(self.medicines)} medicines "
              f"({len(self.searchable_texts)} searchable entries).")

    def _build_index(self):
        """
        For each medicine, we index: its name, every alias, and its uses.
        This means a query like "fever" can match Paracetamol even though
        "fever" never appears in the medicine's name.
        """
        searchable_texts = []
        text_to_medicine = []  # parallel array: which medicine each text belongs to

        for med in self.medicines:
            entries = [med["name"]] + med.get("aliases", []) + med.get("uses", [])
            for entry in entries:
                searchable_texts.append(entry)
                text_to_medicine.append(med)

        embeddings = self.model.encode(searchable_texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Inner product on normalized vectors = cosine similarity
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return index, searchable_texts, text_to_medicine

    def search(self, query: str, top_k: int = 5):
        """
        Returns a list of (medicine_dict, matched_text, score) sorted by
        best match first. Score is cosine similarity, roughly 0 to 1.
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype="float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        seen_medicine_ids = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            med = self.text_to_medicine[idx]
            # Avoid returning the same medicine multiple times (it may have
            # matched via its name AND one of its uses)
            if med["id"] in seen_medicine_ids:
                continue
            seen_medicine_ids.add(med["id"])
            results.append((med, self.searchable_texts[idx], float(score)))

        return results

    def best_match(self, query: str, threshold: float = 0.45):
        """Returns (medicine, score) for the single best match, or (None, 0)."""
        results = self.search(query, top_k=5)
        if not results:
            return None, 0.0
        med, matched_text, score = results[0]
        if score < threshold:
            return None, score
        return med, score
