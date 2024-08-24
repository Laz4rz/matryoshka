# matryoshka

```
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
```

### Outputs
![alt text](images/outputs.png)

**model(x) output:**
```
model_output is a tuple
model_output[0] -- embedding per token -> [batch, tokens, 384]
model_output[1] -- embedding per token + pooling -> [batch, 384]
```

**model pooler:**
```
(pooler): BertPooler(
    (dense): Linear(in_features=384, out_features=384, bias=True)
    (activation): Tanh()
)
```

Instead of pooling in classical sense (aggregation statistic), it takes the CLS token from first postion per each batch item. 

### Matryoshka-Adaptor: Unsupervised and Supervised Tuning for Smaller Embedding Dimensions

##### Losses 

