from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class GraphState(TypedDict):
    question: str
    context: str
    sources: List[str]
    result: dict

def build_graph(retriever, gemini_model):
    def input_node(state: GraphState):
        q = (state.get("question") or "").strip()
        if len(q) < 3:
            raise ValueError("Pregunta demasiado corta")
        return {"question": q}

    def retrieval_node(state: GraphState):
        context, sources, _docs = retriever.get_context(state["question"])
        return {"context": context, "sources": sources}

    def generation_node(state: GraphState):
        from .generation import generate_answer
        result = generate_answer(gemini_model, state["question"], state["context"], state["sources"])
        print(">>> generation_node result:", result)  # debug
        return {"result": result}

    def output_node(state: GraphState):
        return state["result"]

    g = StateGraph(GraphState)
    g.add_node("input", input_node)
    g.add_node("retrieval", retrieval_node)
    g.add_node("generation", generation_node)
    g.add_node("output", output_node)

    g.set_entry_point("input")
    g.add_edge("input", "retrieval")
    g.add_edge("retrieval", "generation")
    g.add_edge("generation", "output")
    g.add_edge("output", END)
    return g.compile()
