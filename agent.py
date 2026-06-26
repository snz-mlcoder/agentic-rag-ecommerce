# agent.py — Layer 3: agentic Corrective-RAG (router + grade + correction loop)
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="./chroma_db",
                     embedding_function=embeddings, collection_name="products")
llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=1024)

MAX_ATTEMPTS = 2  # bound the correction loop -> bounded cost

# ---------- structured-output schemas (force clean, typed decisions) ----------
class RouteDecision(BaseModel):
    destination: Literal["product", "other"] = Field(
        description="'product' if it's a shopping question about electronics "
                    "(headphones, earbuds, speakers, chargers, cables, accessories); "
                    "'other' for anything else.")

class RelevanceGrade(BaseModel):
    relevant: bool = Field(
        description="True only if at least one retrieved product genuinely matches the question.")

router_llm = llm.with_structured_output(RouteDecision)
grader_llm = llm.with_structured_output(RelevanceGrade)

# ---------- State ----------
class State(TypedDict):
    question: str
    query: str
    documents: List[Document]
    relevant: bool
    attempts: int
    route: str
    answer: str

# ---------- nodes ----------
def route_question(state: State):
    d = router_llm.invoke([
        SystemMessage(content=(
            "Decide if the user's message is a shopping question we can answer from our "
            "store catalog. Our catalog contains ONLY these product types: "
            "over-ear noise-cancelling headphones, wireless earbuds, bluetooth speakers, "
            "USB-C charging cables, fast wall chargers, wired earphones, headphone carrying "
            "cases, and phone stands. "
            "Reply 'product' if the message is about buying, comparing, or asking about any "
            "of these (or close accessories). Reply 'other' only for clearly unrelated "
            "messages (general knowledge, weather, chit-chat)."
        )),
        HumanMessage(content=state["question"]),
    ])
    print(f"🧭 router: {d.destination}")
    return {"route": d.destination, "query": state["question"], "attempts": 0}

def retrieve(state: State):
    docs = vectorstore.similarity_search(state["query"], k=4)
    print(f"🔎 retrieve (query={state['query']!r}): {len(docs)} docs")
    return {"documents": docs}

def grade_documents(state: State):
    ctx = "\n\n".join(d.page_content for d in state["documents"])
    g = grader_llm.invoke([
        SystemMessage(content="Grade if the retrieved products are relevant enough to answer."),
        HumanMessage(content=f"Question: {state['question']}\n\nRetrieved:\n{ctx}"),
    ])
    print(f"✅ grade: relevant={g.relevant}")
    return {"relevant": g.relevant}

def transform_query(state: State):
    r = llm.invoke([
        SystemMessage(content="Rewrite this into a better product-search query "
                              "(product type + key attributes). Return only the query."),
        HumanMessage(content=state["query"]),
    ])
    new_q = r.content.strip()
    print(f"♻️  transform: {state['query']!r} -> {new_q!r}")
    return {"query": new_q, "attempts": state["attempts"] + 1}

def generate(state: State):
    ctx = "\n\n---\n\n".join(d.page_content for d in state["documents"])
    sys = SystemMessage(content=(
        "You are a helpful e-commerce shopping assistant. Answer using ONLY the products "
        "below. Recommend specific products by name. If none truly fit, say honestly that "
        "we don't have a matching product — never invent products.\n\n"
        f"AVAILABLE PRODUCTS:\n{ctx}"))
    print("✍️  generate")
    return {"answer": llm.invoke([sys, HumanMessage(content=state["question"])]).content}

def out_of_scope(state: State):
    print("🚫 out_of_scope")
    return {"answer": ("I'm a shopping assistant for our electronics store, so I can only help "
                       "with products like headphones, earbuds, speakers, chargers and accessories.")}

# ---------- conditional logic ----------
def decide_route(state: State):
    return "retrieve" if state["route"] == "product" else "out_of_scope"

def decide_after_grade(state: State):
    if state["relevant"] or state["attempts"] >= MAX_ATTEMPTS:
        return "generate"           # good docs, OR gave up correcting -> honest answer
    return "transform_query"        # weak docs -> rewrite & retry

# ---------- build graph ----------
b = StateGraph(State)
for name, fn in [("route_question", route_question), ("retrieve", retrieve),
                 ("grade_documents", grade_documents), ("transform_query", transform_query),
                 ("generate", generate), ("out_of_scope", out_of_scope)]:
    b.add_node(name, fn)
b.add_edge(START, "route_question")
b.add_conditional_edges("route_question", decide_route,
                        {"retrieve": "retrieve", "out_of_scope": "out_of_scope"})
b.add_edge("retrieve", "grade_documents")
b.add_conditional_edges("grade_documents", decide_after_grade,
                        {"generate": "generate", "transform_query": "transform_query"})
b.add_edge("transform_query", "retrieve")
b.add_edge("generate", END)
b.add_edge("out_of_scope", END)
graph = b.compile()

# ---------- try it on 3 different paths ----------
if __name__ == "__main__":
    for q in [
        "wireless headphones to block noise on flights",   # -> product, direct
        "do you sell a stand for my phone?",               # -> product
        "what's the weather in Venice today?",             # -> out_of_scope
    ]:
        print(f"\n{'='*60}\nQ: {q}")
        out = graph.invoke({"question": q})
        print(f"\nANSWER:\n{out['answer']}\n")