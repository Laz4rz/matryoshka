import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel


class Pooler(nn.Module):

    def __init__(self, hidden_size=None, base_model=None, cls_token_pooler=False, mean_pooler=True):
        super(Pooler, self).__init__()
        if cls_token_pooler:
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.Tanh()
            if base_model is not None:
                with torch.no_grad():
                    weights = base_model.model.pooler.dense.weight[:
                                                                   hidden_size, :
                                                                   hidden_size]
                    biases = base_model.model.pooler.dense.bias[:hidden_size]
                    self.dense.weight.copy_(weights)
                    self.dense.bias.copy_(biases)
            self.pooler = self.cls_token
        elif mean_pooler:
            self.pooler = self.mean_pooler

    def mean_pooler(self, outputs, attention_mask, skip_cls_token=False):
        if skip_cls_token:
            outputs = outputs[:, 1:]
            attention_mask = attention_mask[:, 1:]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            outputs.size()).float()
        sum_embeddings = torch.sum(outputs * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        mean_pooled = sum_embeddings / torch.clamp(sum_mask, min=1e-9)
        return mean_pooled

    def cls_token(self, inputs):
        cls_tokens = inputs[:, 0]
        pooled_output = self.dense(cls_tokens)
        pooled_output = self.activation(pooled_output)
        return pooled_output

    def forward(self, outputs, attention_mask):
        return self.pooler(outputs, attention_mask)


class Normalizer(nn.Module):

    def __init__(self, p=2, dim=1):
        super(Normalizer, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, inputs):
        return F.normalize(inputs, p=self.p, dim=self.dim)


class BaseModel(nn.Module):

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super(BaseModel, self).__init__()
        model = AutoModel.from_pretrained(model_name)
        self.config = model.config
        self.model = model

    def forward(self, **kwargs):
        return self.model(**kwargs)[0] # only return the hidden states, no pooler output


class Matryoshka(nn.Module):

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        matryoshka_dim=64,
    ):
        super(Matryoshka, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = BaseModel(model_name)
        self.pooler = Pooler(hidden_size=matryoshka_dim, base_model=None, mean_pooler=True)
        self.normalizer = Normalizer()

        assert self.model.config.hidden_size >= matryoshka_dim, \
            f"Model hidden size ({self.model.config.hidden_size}) must be greater than or equal to matryoshka_dim ({matryoshka_dim})"

        self.matryoshka_dim = matryoshka_dim

    def encode(self, sentences: list[str],
               **kwargs) -> torch.Tensor | np.ndarray:
        inputs = self.tokenizer(sentences,
                                    padding=True,
                                    truncation=True,
                                    return_tensors='pt',
                                    **kwargs)
        with torch.no_grad():
            matryoshka_output = self(**inputs)
        return matryoshka_output.cpu().numpy()

    def forward(self, **kwargs):
        model_output = self.model(
            **kwargs)
        model_output_reduced = model_output[:, :, :self.matryoshka_dim]
        pooled_output = self.pooler(model_output_reduced, kwargs['attention_mask'])
        normalized_output = self.normalizer(pooled_output)
        return normalized_output
