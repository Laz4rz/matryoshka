from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np

from matryoshka import Matryoshka

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']
sentences = ["sentence", "not a sentence"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
matryoshka = Matryoshka(adaptor=False, matryoshka_dim=384)
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
print("Encoded input shape:", encoded_input['input_ids'].shape)

# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
    matryoshka_output = matryoshka(pooling=False, **encoded_input)

assert torch.allclose(model_output[0], matryoshka_output, atol=1e-6)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

print("Sentence embeddings:")
print(np.dot(sentence_embeddings, sentence_embeddings.T))
