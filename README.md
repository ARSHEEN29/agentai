# 🎓 AI Agent Study Assistant

An autonomous, secure, and production-tested AI Study Assistant agent built with **Python**, **Streamlit**, and the **Google Gen AI SDK**. 

The assistant provides safe arithmetic computation via Python AST, factual retrieval from a local machine-learning study-notes knowledge base, automatic function calling, and comprehensive error handling designed for seamless demonstrations on Windows.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Technology Stack](#technology-stack)
5. [Folder Structure](#folder-structure)
6. [Prerequisites](#prerequisites)
7. [Windows PowerShell Setup & Installation](#windows-powershell-setup--installation)
8. [Environment Configuration](#environment-configuration)
9. [Running the Application](#running-the-application)
10. [Running Automated Tests](#running-automated-tests)
11. [Sample Demonstration Questions](#sample-demonstration-questions)
12. [Root Cause Analysis & Troubleshooting](#root-cause-analysis--troubleshooting)
13. [Viva / Demonstration Guide](#viva--demonstration-guide)

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

## Root Cause Analysis & Troubleshooting

### Why Did `gemini-3.8-flash` Experience 133s Latency or 504 Deadlines?

During Phase 1 diagnosis, direct benchmarks isolated the exact root cause:

1. **Server-Side Queueing on Experimental Tier:**
   - Empirical diagnostic testing showed that `gemini-3.8-flash` requests on free-tier API quotas take between **122.15s and 148.58s** on Google's backend.
   - When HTTP timeout is configured to 30s or 120s, requests abort with `504 DEADLINE_EXCEEDED`.
2. **Model Availability & Deprecation:**
   - Legacy models like `gemini-2.5-flash` return `404 NOT_FOUND` for new API keys with the message: *"This model is no longer available to new users"*.
3. **The Solution:**
   - Selecting `gemini-flash-lite-latest` drops latency from **122+ seconds down to 1.5 - 1.9 seconds** while maintaining 100% tool-calling precision.
   - The UI includes a model selector in the sidebar so users can seamlessly switch models or benchmark them in real time.

### Troubleshooting Common Errors

- **`504 DEADLINE_EXCEEDED`**:
  - The model took too long to answer. In the sidebar, ensure `gemini-flash-lite-latest` is selected, or increase `GEMINI_TIMEOUT_SECONDS=60` in `.env`.
- **`429 RESOURCE_EXHAUSTED`**:
  - Free tier rate limit reached (typically 15 requests/minute). Wait 10 seconds before asking another question.
- **`401 UNAUTHENTICATED / API_KEY_INVALID`**:
  - Your Gemini API key is missing or incorrect. Verify `.env` has `GEMINI_API_KEY=AIzaSy...`.
- **`404 NOT_FOUND`**:
  - The model name is retired or misspelled. Use `gemini-flash-lite-latest`.

---

## Viva / Demonstration Guide

### 1. What is an AI Agent?
An AI Agent is a system where an LLM is given reasoning instructions and access to executable **tools** (functions). Instead of predicting answers directly from memorized training weights, the agent decides *when* and *which* tool to call, executes it, and synthesizes the verified output for the user.

### 2. Why use Python AST instead of `eval()`?
`eval()` executes arbitrary Python code with full system privileges, allowing malicious prompts to execute `__import__('os').system('rmdir /s /q ...')` or steal files. Python's Abstract Syntax Tree (`ast.parse`) parses expressions into a syntax tree, allowing our code to strictly whitelist mathematical operations and reject any unauthorized node.

### 3. How does the agent prevent hallucination for course notes?
When a query asks about notes, the `search_notes` tool checks specific machine learning topic patterns. If the topic is missing, it explicitly returns `NO_MATCH`. The system instructions strictly bind the model to report missing topics rather than inventing facts.

### 4. How does the Google Gen AI SDK handle Automatic Function Calling (AFC)?
With `client.chats.create(tools=[...])`, the SDK automatically manages the conversation cycle:
1. User prompt sent to Gemini.
2. Gemini returns a tool-call request (`function_call`).
3. SDK executes the local Python function and sends the `function_response` back to Gemini.
4. Gemini summarizes the tool output and returns the final human-readable response.
