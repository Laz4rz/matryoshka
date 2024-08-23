import mteb
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)
tasks = mteb.get_tasks(tasks=["Banking77Classification"])
evaluation = mteb.MTEB(tasks=tasks)

print(f"Running evaluation for {model_name}")
results = evaluation.run(model, output_folder=f"results/{model_name}")

print(results)
