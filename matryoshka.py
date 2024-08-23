import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Pooler(nn.Module):

    def __init__(self, hidden_size):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, inputs):
        cls_tokens = inputs[:, 0]
        pooled_output = self.dense(cls_tokens)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class Matryoshka(nn.Module):

    def __init__(self,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 matryoshka_dim=None,
                 raw_pooler=True):
        super(Matryoshka, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.matryoshka_dim = model.config.hidden_size if matryoshka_dim is None else matryoshka_dim
        self.matryoshka = False

        assert self.model.config.hidden_size >= matryoshka_dim, \
            f"Model hidden size ({self.model.config.hidden_size}) must be greater than or equal to matryoshka_dim ({matryoshka_dim})"
        if self.model.config.hidden_size > matryoshka_dim:
            self.matryoshka_dim = matryoshka_dim
            self.matryoshka = True
            if raw_pooler:
                self.pooler = Pooler(self.matryoshka_dim)
            else:
                # this needs some smart matrix dim reduction
                # i can do it stupid but dunno if I should
                self.pooler = self.model.pooler

    def forward(self, **kwargs):
        base_outputs = self.model(**kwargs)
        if self.matryoshka:
            base_outputs["last_hidden_state"] = base_outputs[
                "last_hidden_state"][:, :, :self.matryoshka_dim]
            base_outputs["pooler_output"] = self.pooler(
                base_outputs["last_hidden_state"])
        print("Matryoshka output shapes:",
              base_outputs["last_hidden_state"].shape,
              base_outputs["pooler_output"].shape)
        return base_outputs
