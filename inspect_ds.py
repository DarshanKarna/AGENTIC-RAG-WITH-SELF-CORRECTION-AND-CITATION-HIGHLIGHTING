from datasets import load_dataset

ds = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")
print(ds)
try:
    print(ds["passages"][0])
except:
    pass
