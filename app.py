import collections
import json
import uuid

from flask import Flask, request, render_template, Response, stream_with_context
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}"

PRIMARY_MODEL_NAME   = "gemini-3.6-flash"
FALLBACK_MODEL_NAME  = "gemini-3.1-flash-lite"

_primary_llm  = ChatGoogleGenerativeAI(model=PRIMARY_MODEL_NAME)
_fallback_llm = ChatGoogleGenerativeAI(model=FALLBACK_MODEL_NAME)
embedder      = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# with_fallbacks routes to the lite model automatically when RPD is exhausted
model = _primary_llm.with_fallbacks(
    [_fallback_llm],
    exceptions_to_handle=(ResourceExhausted,),
)

_sqlite_cm = SqliteSaver.from_conn_string("checkpoints.db")
checkpointer = _sqlite_cm.__enter__()

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Never use em dashes (the character -) in any response under any circumstances. "
    "Use a regular hyphen (-) or rewrite the sentence instead."
)

# Primary agent (uses model-with-fallback internally)
agent = create_react_agent(
    model,
    tools=[get_weather],
    checkpointer=checkpointer,
    prompt=SYSTEM_PROMPT,
)

# Explicit fallback agent used when the whole primary stream fails at the outer level
fallback_agent = create_react_agent(
    _fallback_llm,
    tools=[get_weather],
    checkpointer=checkpointer,
    prompt=SYSTEM_PROMPT,
)

app = Flask(__name__)

call_counts = {}

CACHE_SIMILARITY_THRESHOLD = 0.95
CACHE_MAX_SIZE = 300
SEMANTIC_CACHE: collections.deque = collections.deque(maxlen=CACHE_MAX_SIZE)


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def find_cached_answer(query_vector):
    best_score, best_answer = 0.0, None
    for vec, _q, answer in SEMANTIC_CACHE:
        score = cosine_similarity(query_vector, vec)
        if score > best_score:
            best_score, best_answer = score, answer
    if best_score >= CACHE_SIMILARITY_THRESHOLD:
        return best_answer
    return None


def store_in_cache(query_vector, question, answer):
    SEMANTIC_CACHE.append((query_vector, question, answer))


def extract_text(content):
    """Handles both plain string chunks and Gemini's list-of-parts content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


@app.route("/")
def index():
    session_id = str(uuid.uuid4())
    call_counts[session_id] = 0
    return render_template("index.html", session_id=session_id)


@app.route("/chat-stream", methods=["POST"])
def chat_stream():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    message = data.get("message", "").strip()
    if not session_id or not message:
        return Response(
            f"data: {json.dumps({'type': 'error', 'message': 'Missing session_id or message'})}\n\n",
            mimetype="text/event-stream",
        )
    config = {"configurable": {"thread_id": session_id}}

    def _stream_agent(active_agent, model_name, full_text_ref):
        """Shared streaming logic for both primary and fallback agents."""
        yield f"data: {json.dumps({'type': 'model_info', 'model': model_name})}\n\n"
        for stream_mode, chunk in active_agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config,
            stream_mode=["updates", "messages"],
        ):
            if stream_mode == "updates":
                for node_name, node_output in chunk.items():
                    if node_name != "model":
                        continue
                    for m in node_output.get("messages", []):
                        tool_calls = getattr(m, "tool_calls", None)
                        if not tool_calls:
                            continue
                        for tc in tool_calls:
                            payload = {
                                "type": "tool_call",
                                "name": tc.get("name"),
                                "args": tc.get("args"),
                            }
                            yield f"data: {json.dumps(payload)}\n\n"

            elif stream_mode == "messages":
                message_chunk, metadata = chunk
                if metadata.get("langgraph_node") != "model":
                    continue
                text = extract_text(message_chunk.content)
                if text:
                    full_text_ref.append(text)
                    yield f"data: {json.dumps({'type': 'token', 'text': text})}\n\n"

    def generate():
        try:
            query_vector = embedder.embed_query(message)
            cached_answer = find_cached_answer(query_vector)

            if cached_answer is not None:
                yield f"data: {json.dumps({'type': 'cache_hit'})}\n\n"
                for word in cached_answer.split(" "):
                    yield f"data: {json.dumps({'type': 'token', 'text': word + ' '})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'call_count': call_counts.get(session_id, 0)})}\n\n"
                return

            full_text_parts = []

            try:
                yield from _stream_agent(agent, PRIMARY_MODEL_NAME, full_text_parts)
            except ResourceExhausted:
                # Primary model RPD exhausted — fall back to lite model
                full_text_parts.clear()
                yield f"data: {json.dumps({'type': 'fallback'})}\n\n"
                yield from _stream_agent(fallback_agent, FALLBACK_MODEL_NAME, full_text_parts)

            full_text = "".join(full_text_parts)
            if full_text.strip():
                store_in_cache(query_vector, message, full_text)

            call_counts[session_id] = call_counts.get(session_id, 0) + 1
            yield f"data: {json.dumps({'type': 'done', 'call_count': call_counts[session_id]})}\n\n"

        except Exception as exc:  # noqa: BLE001
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, port=port, threaded=True)