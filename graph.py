# graph.py — Layer 2: first LangGraph RAG graph (retrieve -> generate)
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

# 1. load the vector store built in Layer 1.
#    ⚠️ MUST be the SAME embedding model as ingest, or search breaks silently.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="products",
)

# 2. the LLM — Haiku: cheap, fast, fine for grounded generation
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)

# 3. the State that flows through the graph
class State(TypedDict):
    question: str
    documents: List[Document]
    answer: str

# 4. node: retrieve relevant products
def retrieve(state: State):
    docs = vectorstore.similarity_search(state["question"], k=4)
    return {"documents": docs}

# 5. node: generate a grounded answer
def generate(state: State):
    context = "\n\n---\n\n".join(d.page_content for d in state["documents"])
    system = SystemMessage(content=(
        "You are a helpful e-commerce shopping assistant. "
        "Answer the customer's question using ONLY the products listed below. "
        "Recommend specific products by name. If the products don't cover what "
        "they asked, say so honestly — never invent products.\n\n"
        f"AVAILABLE PRODUCTS:\n{context}"
    ))
    response = llm.invoke([system, HumanMessage(content=state["question"])])
    return {"answer": response.content}

# 6. build + compile the graph
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# 7. try it
if __name__ == "__main__":
    question = "I need wireless headphones that block out background noise for flights."
    result = graph.invoke({"question": question})
    print("Q:", question)
    print("\nRETRIEVED:")
    for d in result["documents"]:
        print("  -", d.metadata["title"])
    print("\nANSWER:\n", result["answer"])