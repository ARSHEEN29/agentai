# 🎓 AI Agent Study Assistant

An autonomous, secure, and production-tested AI Study Assistant agent built with **Python**, **Streamlit**, and the **Google Gen AI SDK**. 

The assistant provides safe arithmetic computation via Python AST, factual retrieval from a local machine-learning study-notes knowledge base, automatic function calling, and comprehensive error handling designed for seamless demonstrations on Windows.

---

## Project Overview

The **AI Agent Study Assistant** is designed for college students and educators to study core machine-learning concepts and perform mathematical operations without hallucination. 

Traditional LLMs often struggle with precise arithmetic and can fabricate facts when answering questions beyond their training data. This agent solves those issues using **Agentic Tool Calling**:
- Arithmetic expressions are intercepted and computed deterministically by a secure AST calculator.
- Questions regarding study topics are retrieved directly from verified course notes.
- Queries outside the curriculum are accurately flagged without hallucination.

---

## Key Features

- **Safe AST Calculator (Tool A):**
  - Evaluates addition (`+`), subtraction (`-`), multiplication (`*`), division (`/`), modulo (`%`), power (`**`), and unary operators (`+`, `-`).
  - Evaluates expressions strictly using Python Abstract Syntax Trees (`ast`). **Never uses `eval()`**.
  - Safe guards against division by zero, non-finite results (`NaN`/`Inf`), oversized exponents (`abs(exp) > 100`), and excessive string lengths.
  - Rejects arbitrary code execution, module imports, and system calls.

- **Study Notes Knowledge Base (Tool B):**
  - Contains verified definitions for five foundational ML topics:
    1. *Supervised learning*
    2. *Unsupervised learning*
    3. *Overfitting*
    4. *Classification*
    5. *Regression*
  - Uses regex word-boundary pattern matching to avoid false positives on generic queries (e.g. *"What is machine learning?"* or *"What is learning?"*).
  - Explicitly returns `NO_MATCH` for unsupported topics (e.g., *"Quantum computing"* or *"Blockchain"*).

- **High-Performance Tool Calling:**
  - Integrated via Google Gen AI SDK (`google-genai` version 2.33.0+) utilizing `client.chats.create` for native Automatic Function Calling (AFC).
  - Sub-2 second response latency using `gemini-flash-lite-latest`.

- **Production Error Handling & Diagnostics:**
  - Granular classification for HTTP 504 (timeout), 429 (rate limits), 401/403 (authentication), 404 (model retired), and network disconnects.
  - Interactive Streamlit sidebar with live model selection, quick example buttons, topic list, and conversation reset.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │    User / Web Browser   │
                    └───────────┬─────────────┘
                                │ Question / Calculation
                                ▼
                    ┌─────────────────────────┐
                    │  Streamlit UI (app.py)  │
                    └───────────┬─────────────┘
                                │ Chat Session
                                ▼
                    ┌─────────────────────────┐
                    │ Google Gen AI SDK Agent │
                    │ (gemini-flash-lite)     │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼ Tool Call                                     ▼ Tool Call
┌──────────────────────────────┐        ┌──────────────────────────────┐
│     AST Calculator Tool      │        │   Study Notes Search Tool    │
│  - ast.parse (whitelist ops) │        │  - Word-boundary regex       │
│  - ZeroDivisionError guard   │        │  - 5 core ML definitions     │
│  - Overflow/Exponent guard   │        │  - Explicit NO_MATCH guard   │
│  - No eval() execution       │        │  - Hallucination-free        │
└──────────────┬───────────────┘        └──────────────┬───────────────┘
               │ Result                                │ Definition / NO_MATCH
               └───────────────────────┬───────────────┘
                                       │
                                       ▼
                    ┌─────────────────────────┐
                    │ Synthesized Answer + UI │
                    │  (Response Time Tagged) │
                    └─────────────────────────┘
```

---

## Technology Stack

- **Language:** Python 3.10+ (tested on Python 3.14 on Windows)
- **Frontend / Web UI:** Streamlit (`streamlit>=1.55.0`)
- **AI SDK:** Google Gen AI SDK (`google-genai>=2.33.0`)
- **Configuration & Secrets:** `python-dotenv>=1.2.0`
- **Parsing & AST:** Python built-in `ast`, `operator`, `math`, `re`
- **Automated Testing:** Python standard `unittest` and `unittest.mock`

---

## Folder Structure

```text
agentbuild-study-assistant/
├── app.py              # Main Streamlit web application & agent logic
├── tests.py            # Comprehensive automated test suite (40 unit tests)
├── test_model.py       # Live API connectivity & latency diagnostic benchmark
├── requirements.txt    # Application dependencies
├── .env.example        # Environment variable template (safe, no secrets)
├── .env                # Local secrets (never committed to git)
├── .gitignore          # Excludes .env, .venv, caches, and secrets
└── README.md           # Full technical documentation & demo guide
```

---

## Prerequisites

1. **Windows 10/11** with PowerShell.
2. **Python 3.10 or higher** installed and added to your `PATH`.
3. A free **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).

---

## Windows PowerShell Setup & Installation

Open Windows PowerShell, navigate to the project directory, and run:

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# If script execution is restricted on Windows, allow local scripts:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Environment Configuration

1. Copy the example configuration to create your local `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Open `.env` in any text editor and configure your credentials:
   ```env
   # Your Google Gemini API Key
   GEMINI_API_KEY=AIzaSy...your_actual_key_here

   # Recommended fast model (responds in ~1.5 seconds)
   GEMINI_MODEL=gemini-flash-lite-latest

   # HTTP Timeout in seconds (default: 30)
   GEMINI_TIMEOUT_SECONDS=30
   ```

> [!CAUTION]
> Never commit `.env` to version control or display your API key during screen recordings. The `.gitignore` file automatically excludes `.env`.

---

## Running the Application

Launch the Streamlit web application:

```powershell
python -m streamlit run app.py
```

Streamlit will display the local URL:
```text
  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```
Open `http://localhost:8501` in your browser.

---

## Running Automated Tests

Run the full automated test suite (40 tests covering AST arithmetic, boundary guards, notes retrieval, false match prevention, missing key handling, error formatting, and mocked tool calling):

```powershell
python -m unittest tests.py -v
```

All 40 unit tests execute in **under 0.1 seconds** and **do not require an active API key or internet access**.

To test live Gemini API connectivity and latency:

```powershell
python test_model.py
```

---

## Sample Demonstration Questions

| Category | Example Question | Expected Tool Call | Expected Behavior |
| :--- | :--- | :--- | :--- |
| **Arithmetic** | `What is 25 * 16?` | `calculator("25 * 16")` | Computes exactly `400` without estimation. |
| **Study Notes** | `What is overfitting according to the notes?` | `search_notes("overfitting")` | Returns: *"Overfitting occurs when a model fits training data too closely and performs poorly on unseen data."* |
| **Study Notes** | `Define classification.` | `search_notes("classification")` | Returns: *"Classification is a supervised learning task that predicts a category or class."* |
| **Missing Topic**| `What do the notes say about quantum computing?`| `search_notes("quantum computing")` | Returns `NO_MATCH`. Agent explains the topic is not in the notes without fabricating facts. |
| **Error Guard** | `What is 100 / 0?` | `calculator("100 / 0")` | Returns `ERROR: Division by zero.` Agent clearly explains division by zero is mathematically undefined. |

---

