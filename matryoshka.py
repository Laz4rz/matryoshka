import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel


MODEL_CARD_DATA = {
    "name": "Matryoshka",
    "description": "This is a custom model trained for text embedding with dimensionality reduction.",
    "version": "1.0",
    "license": "Apache-2.0",
    "limitations": "Not suitable for legal or medical advice.",
    "author": "Laz4rz",
    "contact": "laz4rz@gmail",
}


class Pooler(nn.Module):

    def __init__(self, hidden_size=None, base_model=None, cls_token_pooler=False, mean_pooler=True):
        super(Pooler, self).__init__()
        if cls_token_pooler:
            self.dense = nn.Linear(hidden_size, hidden_size)
            self.activation = nn.Tanh()
            if base_model is not None:
                with torch.no_grad():
                    weights = base_model.model.pooler.dense.weight[:hidden_size, :hidden_size]
                    biases = base_model.model.pooler.dense.bias[:hidden_size]
                    self.dense.weight.copy_(weights)
                    self.dense.bias.copy_(biases)
            self.pooler = self.cls_token_pooler
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

    def cls_token_pooler(self, inputs):
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
    

class Adaptor(nn.Module):
    
        def __init__(self, hidden_size):
            super(Adaptor, self).__init__()
            self.down_project = nn.Linear(hidden_size, 256)
            self.activation = nn.ReLU()
            self.up_project = nn.Linear(256, hidden_size)

        def forward(self, inputs):
            down_projected = self.activation(self.down_project(inputs))
            up_projected = self.up_project(down_projected)
            return up_projected


class Matryoshka(nn.Module):

    def __init__(self,
                 model_name="sentence-transformers/all-MiniLM-L6-v2",
                 matryoshka_dim=64,
                 device="cpu",
                 adaptor=False,
                 disable_gradients=True
                 ):
        super(Matryoshka, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = BaseModel(model_name)
        self.pooler = Pooler(hidden_size=matryoshka_dim,
                             base_model=None,
                             mean_pooler=True)
        self.normalizer = Normalizer()

        assert self.model.config.hidden_size >= matryoshka_dim, \
            f"Model hidden size ({self.model.config.hidden_size}) must be greater than or equal to matryoshka_dim ({matryoshka_dim})"
        self.name = f"Matryoshka(model={model_name.split('/')[-1]}, dim={matryoshka_dim})"
        self.model_card_data = MODEL_CARD_DATA
        self.matryoshka_dim = matryoshka_dim
        self.adaptor = adaptor
        if disable_gradients:
            for param in self.model.parameters():
                param.requires_grad = False
        if adaptor:
            self.adaptor = Adaptor(self.model.config.hidden_size)

    def encode(self, sentences, batch_size=32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        all_embeddings = []

        valid_kwargs = {}
        for key in kwargs:
            if key in self.tokenizer.model_input_names:
                valid_kwargs[key] = kwargs[key]

        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in range(0, len(sentences), batch_size):
            sentences_batch = sentences_sorted[start_index:start_index +
                                               batch_size]

            real_batch_size = len(sentences_batch)

            inputs = self.tokenizer(sentences_batch,
                                    padding=True,
                                    truncation=True,
                                    return_tensors="pt",
                                    **valid_kwargs)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings = self(**inputs)

            assert embeddings.shape == (
                real_batch_size, self.matryoshka_dim
            ), f"{start_index}: Expected shape ({batch_size}, {self.matryoshka_dim}), got {embeddings.shape}"

            all_embeddings.extend(embeddings.cpu().numpy())

        all_embeddings = [
            all_embeddings[idx] for idx in np.argsort(length_sorted_idx)
        ]

        return np.array(all_embeddings)

    def forward(self, pooling=True, reduce=True, **kwargs):
        output = self.model(**kwargs)

        if self.adaptor:
            output = self.adaptor(output)

        if reduce:
            output = output[:, :, :self.matryoshka_dim]

        if pooling:
            output = self.pooler(output,
                                        kwargs['attention_mask'])
            output = self.normalizer(output)
        return output
