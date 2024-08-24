# matryoshka

```
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
```

### Outputs
![alt text](images/outputs.png)

**model(x) output:**
```python
model_output is a tuple
model_output[0] -- embedding per token -> [batch, tokens, 384]
model_output[1] -- embedding per token + pooling -> [batch, 384]
```

**model pooler:**
```python
(pooler): BertPooler(
    (dense): Linear(in_features=384, out_features=384, bias=True)
    (activation): Tanh()
)
```

Instead of pooling in classical sense (aggregation statistic), it takes the CLS token from first postion per each batch item. 


### BEIR

BEIR data organization is an eldritchian horror. If you look at it in a HuggingFace viewer -- it doesn't make any sense. You have queries and text "passages", that you're suppossed to retrieve. But how do they link? What's going on? It's not there. We don't see it. Turns out the right way is to either download a raw dataset and look inside yourself or use the original BEIR script. 

BEIR script is easier.

```python
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#### Download scifact.zip dataset and unzip the dataset
dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join("data")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

length = None
corpus = {k: v for k, v in list(corpus.items())[:length]}
queries = {k: v for k, v in list(queries.items())[:10]}
qrels = {k: v for k, v in list(qrels.items())[:length]}
```

Here:
 - corpus: text "passages" (sentences) (with ids) that we are suppossed to retrieve
 - queries: we take these and try to match them with correct corpus text passages
 - qrels: the whole gist, this matches ids of queries with ids of corpus texts, each connection is assigned an importance value, 2 for highly relevant, 1 for less relevant, no connection if irrelevant (I think)

### Matryoshka-Adaptor: Unsupervised and Supervised Tuning for Smaller Embedding Dimensions

##### Losses 

