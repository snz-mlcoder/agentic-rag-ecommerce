# ingest.py — Layer 1: load products → embed → store in vector DB
import os, shutil
import pandas as pd
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# rebuild fresh each run (so re-running doesn't create duplicates)
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# 1. load the product catalog
df = pd.read_parquet("data/products.parquet")
print(f"Loaded {len(df)} products")

# 2. turn each product into a Document: text to embed + metadata to keep
def build_text(row):
    return (
        f"Title: {row['product_title']}\n"
        f"Brand: {row['product_brand']}\n"
        f"Description: {row['product_description']}\n"
        f"Features: {row['product_bullet_point']}"
    )

documents = [
    Document(
        page_content=build_text(row),
        metadata={
            "product_id": row["product_id"],
            "title": row["product_title"],
            "brand": row["product_brand"],
            "color": row["product_color"],
        },
    )
    for _, row in df.iterrows()
]

# 3. embedding model — all-MiniLM-L6-v2, runs locally, NO API key, NO cost
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. build + persist the vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="products",
)
print(f"Stored {len(documents)} products in ./chroma_db")

# 5. smoke test — does semantic search return sensible products?
query = "wireless noise cancelling headphones"
for r in vectorstore.similarity_search(query, k=3):
    print(" -", r.metadata["title"])