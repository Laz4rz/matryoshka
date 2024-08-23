import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from matryoshka import Matryoshka

sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
matryoshka = Matryoshka(matryoshka_dim=384)

embeddings = model.encode(sentences)
matryoshka_embeddings = matryoshka.encode(sentences)

print("\nShape comparison")
print(embeddings.shape)
print(matryoshka_embeddings.shape)
print("\nValue comparison")
print(embeddings[0][:5])
print(matryoshka_embeddings[0][:5])
allclose_eps = 1e-6
print(f"allclose ({allclose_eps}):",
      np.allclose(embeddings, matryoshka_embeddings, atol=allclose_eps))
allclose_eps = 1e-7
print(f"allclose ({allclose_eps}):",
      np.allclose(embeddings, matryoshka_embeddings, atol=allclose_eps))
print("\nSimilarity comparison")
print("baseline vs baseline", np.dot(embeddings, embeddings.T))
print("baseline vs matryoshka", np.dot(embeddings, matryoshka_embeddings.T))
