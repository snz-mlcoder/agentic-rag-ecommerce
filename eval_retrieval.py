# eval_retrieval.py — Layer 4a: evaluate retrieval against ESCI ground-truth
import statistics as st
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- a design decision you must be able to defend ---
RELEVANT_LABELS = {"E"}      # ESCI "Exact" = ground-truth relevant
K_VALUES = [1, 4, 8]

# same vector store the agent uses (same embedding model — non-negotiable)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db",
                     embedding_function=embeddings, collection_name="products")

# ground truth: query -> set of relevant product_ids
df = pd.read_parquet("data/examples.parquet")
gt = (df[df["esci_label"].isin(RELEVANT_LABELS)]
        .groupby("query")["product_id"].apply(set).to_dict())
queries = list(gt.keys())
print(f"Evaluating {len(queries)} queries | relevant labels = {RELEVANT_LABELS}\n")

def retrieved_ids(query, k):
    return [d.metadata["product_id"]
            for d in vectorstore.similarity_search(query, k=k)]

print(f"{'k':>3} | {'Recall@k':>9} | {'Precision@k':>11} | {'Hit@k':>6} | {'MRR@k':>6}")
print("-" * 48)
for k in K_VALUES:
    recalls, precisions, hits, rr = [], [], [], []
    for q in queries:
        relevant = gt[q]
        retrieved = retrieved_ids(q, k)
        n_hit = sum(1 for pid in retrieved if pid in relevant)
        recalls.append(n_hit / len(relevant))                # coverage of relevant set
        precisions.append(n_hit / k)                          # cleanliness of top-k
        hits.append(1.0 if n_hit > 0 else 0.0)                # did we fail completely?
        rank = next((i + 1 for i, pid in enumerate(retrieved) if pid in relevant), None)
        rr.append(1.0 / rank if rank else 0.0)                # rank of first relevant
    print(f"{k:>3} | {st.mean(recalls):>9.3f} | {st.mean(precisions):>11.3f} | "
          f"{st.mean(hits):>6.3f} | {st.mean(rr):>6.3f}")