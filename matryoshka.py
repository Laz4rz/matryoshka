import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Pooler(nn.Module):

    def __init__(self, hidden_size=None, base_model=None):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        if base_model is not None:
            with torch.no_grad():
                weigths = model.state_dict()["0.auto_model.pooler.dense.weight"][:hidden_size, :hidden_size]
                biases = model.state_dict()["0.auto_model.pooler.dense.bias"][:hidden_size]
                self.dense.weight.copy_(weights)
                self.dense.bias.copy_(biases)

    def forward(self, inputs):
        cls_tokens = inputs[:, 0]
        pooled_output = self.dense(cls_tokens)
        pooled_output = self.activation(pooled_output)
        return pooled_output


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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, **kwargs):
        return self.model(**kwargs)


class Matryoshka(nn.Module):

    def __init__(self,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 matryoshka_dim=64,
                 raw_pooler=True,
                 only_hidden_states=False):
        assert self.model.config.hidden_size >= matryoshka_dim, \
            f"Model hidden size ({self.model.config.hidden_size}) must be greater than or equal to matryoshka_dim ({matryoshka_dim})"

        super(Matryoshka, self).__init__()

        self.matryoshka_dim = matryoshka_dim

        self.model = BaseModel(model_name)[0]  # not pooled outputs
        self.pooler = Pooler(hidden_size=self.matryoshka_dim,
                             base_model=self.model)
        self.normalizer = Normalizer()

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        pooled_output = self.pooler(model_output)
        normalized_output = self.normalizer(pooled_output)
        return normalized_output
