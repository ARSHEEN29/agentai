"""Study Assistant AI - Streamlit Application

A reliable, beginner-friendly AI Study Assistant agent with:
- Safe AST-based Calculator tool
- Local Study-Notes Search tool
- Google Gen AI SDK tool-calling integration
- Configurable timeout and robust error classification
"""

import ast
import math
import operator
import os
import re
import time
from typing import Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Study Assistant AI",
    page_icon="🤖",
    layout="wide",
)

# ---------------- Knowledge Base ----------------
# Built-in study notes (5 core machine learning concepts)
NOTES = {
    "supervised learning": (
        "Supervised learning trains a model using labeled examples."
    ),
    "unsupervised learning": (
        "Unsupervised learning finds patterns in unlabeled data."
    ),
    "overfitting": (
        "Overfitting occurs when a model fits training data too closely "
        "and performs poorly on unseen data."
    ),
    "classification": (
        "Classification is a supervised learning task that predicts "
        "a category or class."
    ),
    "regression": (
        "Regression is a supervised learning task that predicts "
        "a continuous numerical value."
    ),
}

# Distinguishing keyword regex patterns to prevent false matches from generic words
TOPIC_PATTERNS = {
    "unsupervised learning": [r"\bunsupervised\b"],
    "supervised learning": [r"\bsupervised\b"],
    "overfitting": [r"\boverfit(ting|s)?\b"],
    "classification": [r"\bclassif(ication|y|ier|iers|ying)\b"],
    "regression": [r"\bregress(ion|or|ors|ive)?\b"],
}


# ---------------- Tool A: Safe Calculator ----------------
def calculator(expression: str) -> str:
    """Safely calculate a basic arithmetic expression using +, -, *, /, %, **.
    
    Evaluates arithmetic strictly using Python AST without ever calling eval().
    Handles division by zero, non-finite values, and oversized exponents.
    """
    allowed_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value

        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            left = evaluate(node.left)
            right = evaluate(node.right)

            if isinstance(node.op, ast.Pow):
                if abs(right) > 100 or (abs(left) > 1000 and right > 10):
                    raise ValueError("Exponent or calculation too large.")

            if isinstance(node.op, (ast.Div, ast.Mod)) and right == 0:
                raise ZeroDivisionError("Division by zero.")

            result = allowed_ops[type(node.op)](left, right)

            if isinstance(result, (int, float)) and not math.isfinite(result):
                raise ValueError("Calculation resulted in a non-finite value.")

            return result

        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            return allowed_ops[type(node.op)](evaluate(node.operand))

        raise ValueError("Unsupported syntax or operation.")

    try:
        clean_expr = str(expression).strip()
        if len(clean_expr) > 100:
            return "ERROR: Expression is too long."
        if not clean_expr:
            return "ERROR: Empty expression."

        tree = ast.parse(clean_expr, mode="eval")
        result = evaluate(tree.body)
        return str(result)

    except ZeroDivisionError:
        return "ERROR: Division by zero."
    except OverflowError:
        return "ERROR: Exponent or calculation too large."
    except SyntaxError:
        return "ERROR: Invalid arithmetic syntax."
    except Exception as exc:
        return f"ERROR: {str(exc)}"


# ---------------- Tool B: Study Notes Search ----------------
def search_notes(query: str) -> str:
    """Search the built-in study notes for supported machine learning topics.
    
    Matches specific topic keywords and prevents false matches from generic
    question words like 'learning', 'machine', or 'what'.
    """
    clean_query = str(query).lower().strip()
    if not clean_query:
        return "NO_MATCH: The supplied study notes do not contain this topic."

    matched_topics = []
    for topic, patterns in TOPIC_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, clean_query):
                matched_topics.append(topic)
                break

    if not matched_topics:
        return "NO_MATCH: The supplied study notes do not contain this topic."

    results = [f"Topic: {t}\nNote: {NOTES[t]}" for t in matched_topics]
    return "\n\n".join(results)


# ---------------- Client & Agent Logic ----------------
def get_client(timeout_seconds: Optional[int] = None) -> Optional[genai.Client]:
    """Create the Google Gen AI client with configured timeout.
    
    Reads GEMINI_API_KEY and GEMINI_TIMEOUT_SECONDS from the environment.
    Never exposes or logs secret values.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    if timeout_seconds is None:
        try:
            timeout_seconds = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
        except ValueError:
            timeout_seconds = 30

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_seconds * 1000),
    )


SYSTEM_INSTRUCTION = """You are a helpful and knowledgeable Study Assistant AI agent.

You have access to two callable tools:
1. calculator: Use this tool to compute ANY arithmetic calculations accurately.
2. search_notes: Use this tool to retrieve definitions from the supplied study notes for:
   - supervised learning
   - unsupervised learning
   - overfitting
   - classification
   - regression

Rules:
1. When arithmetic is needed, ALWAYS call calculator instead of estimating.
2. When answering questions about the course topics, ALWAYS query search_notes.
3. If search_notes returns NO_MATCH, explicitly inform the student that the supplied study notes do not contain the requested topic. Do NOT invent facts or claim they came from the notes.
4. If a tool returns an ERROR message, explain the error clearly and helpfully.
5. Keep your responses clear, accurate, and concise.
6. Never claim you used a tool unless you actually called it.
"""


def format_api_error(exc: Exception, model_name: str) -> str:
    """Format exceptions into actionable, friendly error messages."""
    error_text = str(exc)

    if "504" in error_text or "DEADLINE_EXCEEDED" in error_text:
        return (
            f"⏱️ **Timeout / Deadline Exceeded (504)**\n\n"
            f"The model `{model_name}` did not respond within the configured timeout.\n\n"
            f"**Recommendation:** Switch to the fast recommended model "
            f"`gemini-flash-lite-latest` in the sidebar or increase `GEMINI_TIMEOUT_SECONDS`."
        )

    if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
        return (
            "⏳ **Rate Limit / Quota Exceeded (429)**\n\n"
            "The free-tier request rate limit was reached for this model. "
            "Please wait 5-10 seconds before asking your next question."
        )

    if any(k in error_text for k in ["401", "403", "API_KEY_INVALID", "UNAUTHENTICATED", "PERMISSION_DENIED"]):
        return (
            "🔑 **Authentication Failure**\n\n"
            "The Gemini API key is missing, invalid, or lacks permissions. "
            "Please verify `GEMINI_API_KEY` in your `.env` file."
        )

    if "404" in error_text or "NOT_FOUND" in error_text:
        return (
            f"❌ **Model Unavailable (404)**\n\n"
            f"The model `{model_name}` is not available or has been retired for your account tier. "
            f"Please switch to `gemini-flash-lite-latest` in the sidebar."
        )

    if "ConnectionError" in error_text or "ConnectError" in error_text or "getaddrinfo failed" in error_text:
        return (
            "🌐 **Network Connection Error**\n\n"
            "Unable to connect to Google Gemini API servers. "
            "Please check your internet connection and proxy settings."
        )

    return (
        f"⚠️ **Request Error**\n\n"
        f"Could not complete the request.\n\n"
        f"*Details:* `{error_text[:200]}`"
    )


def ask_agent(client: genai.Client, question: str, model_name: str) -> str:
    """Send the question to Gemini using chat-based automatic function calling."""
    chat = client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0,
            max_output_tokens=300,
            tools=[calculator, search_notes],
        ),
    )

    response = chat.send_message(question)
    return response.text or "I completed the request but did not generate a text summary."


# ---------------- Streamlit User Interface ----------------
def main():
    # Sidebar: Project Info, Topics, Models, Settings
    with st.sidebar:
        st.title("🤖 Assistant Config")
        st.write("An autonomous agent with **Safe Calculator** & **Study Notes Search**.")
        st.divider()

        # Model Selector
        env_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        default_model = "gemini-flash-lite-latest"

        options = [
            "gemini-flash-lite-latest (Recommended - Fast)",
            "gemini-3.5-flash-lite (Fast Preview)",
        ]
        if env_model not in ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]:
            options.append(f"{env_model} (.env configured)")

        selected_option = st.selectbox(
            "Active Gemini Model:",
            options=options,
            index=0,
            help="gemini-flash-lite-latest responds in ~1.5s. Models like gemini-3.8-flash may take over 120s on free tier."
        )
        active_model = selected_option.split(" ")[0]

        # Timeout setting display
        timeout_setting = os.getenv("GEMINI_TIMEOUT_SECONDS", "30")
        st.caption(f"⚙️ Configured Timeout: **{timeout_setting}s**")

        st.divider()
        st.subheader("📚 Available Study Topics")
        for topic in NOTES:
            st.markdown(f"- **{topic.title()}**")

        st.divider()
        st.subheader("💡 Example Questions")
        example_prompts = [
            "What is 25 * 16?",
            "What is overfitting according to the notes?",
            "What is classification according to the notes?",
            "What is quantum computing according to the notes?",
            "What is 100 / 0?",
        ]

        for eg in example_prompts:
            if st.button(f"👉 {eg}", key=f"btn_{eg}", use_container_width=True):
                st.session_state["pending_input"] = eg

        st.divider()
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state["pending_input"] = None
            st.rerun()

    # Main Chat View
    st.title("🎓 AI Agent Study Assistant")
    st.markdown(
        "Ask questions about your study notes, perform accurate arithmetic calculations, "
        "or test knowledge boundaries. The agent will call the appropriate tool."
    )

    # API Key check
    client = get_client()
    if client is None:
        st.error(
            "⚠️ **Gemini API Key missing!**\n\n"
            "Please create a `.env` file in the project directory containing:\n"
            "```env\n"
            "GEMINI_API_KEY=your_actual_gemini_api_key\n"
            "GEMINI_MODEL=gemini-flash-lite-latest\n"
            "GEMINI_TIMEOUT_SECONDS=30\n"
            "```\n"
            "Then refresh or restart the application."
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "elapsed" in msg:
                st.caption(f"⏱️ Response time: {msg['elapsed']:.2f} seconds")

    # Handle input from either text box or quick example button
    pending_prompt = st.session_state.pop("pending_input", None)
    user_input = st.chat_input("Ask a question (e.g. 'What is regression?' or '125 / 5')...")
    prompt = pending_prompt or user_input

    if prompt:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            if client is None:
                err_msg = "Please configure your GEMINI_API_KEY in `.env` to start chatting."
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            else:
                with st.spinner(f"Agent thinking with `{active_model}`..."):
                    start_time = time.perf_counter()
                    try:
                        answer = ask_agent(client, prompt, active_model)
                        elapsed = time.perf_counter() - start_time

                        st.markdown(answer)
                        st.caption(f"⏱️ Response time: {elapsed:.2f} seconds · Model: `{active_model}`")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "elapsed": elapsed
                        })

                    except Exception as exc:
                        elapsed = time.perf_counter() - start_time
                        error_formatted = format_api_error(exc, active_model)

                        st.error(error_formatted)
                        st.caption(f"⚠️ Failed after {elapsed:.2f} seconds · Model: `{active_model}`")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_formatted,
                            "elapsed": elapsed
                        })


if __name__ == "__main__":
    main()