from torchview import draw_graph
from transformers import AutoTokenizer, AutoModel

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

model_graph = draw_graph(model, input_size=(1, 10))

