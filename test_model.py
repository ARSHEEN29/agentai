"""Live Model Verification and Latency Diagnostic Tool.

Tests live Gemini API connectivity, response time, and tool calling.
Compares configured model against the fast recommended model.
Never logs or prints secret API keys. Safe on Windows PowerShell.
"""

import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Ensure safe console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
configured_model = os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
try:
    timeout_sec = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
except ValueError:
    timeout_sec = 30

if not api_key:
    print("[ERROR] GEMINI_API_KEY is missing from .env")
    print("Please add your key to .env without quotes or spaces.")
    sys.exit(1)

def run_diagnostic_on_model(target_model: str, client: genai.Client):
    print(f"\n>>> Running diagnostics on model: '{target_model}' <<<")

    # 1. Direct generation test
    print("  [1/2] Direct Generation ('Reply with OK')...")
    start = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=target_model,
            contents="Reply with exactly one word: OK",
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=10
            )
        )
        elapsed = time.perf_counter() - start
        output_text = response.text.strip() if response.text else "(no text returned)"
        print(f"        [STATUS] SUCCESS")
        print(f"        [OUTPUT] {output_text}")
        print(f"        [LATENCY] {elapsed:.2f} seconds")

        if elapsed > 15.0:
            print(f"        [WARN] Model latency is very high ({elapsed:.2f}s).")

    except Exception as error:
        elapsed = time.perf_counter() - start
        print(f"        [FAILED] after {elapsed:.2f} seconds")
        print(f"        [ERROR] {type(error).__name__}: {str(error)[:160]}")
        if "504" in str(error) or "DEADLINE_EXCEEDED" in str(error):
            print("        [ANALYSIS] 504 Deadline Exceeded: Model queueing/overloaded on server.")

    # 2. Tool calling test
    print("  [2/2] Tool-Calling Agent ('What is 25 * 16?')...")
    def dummy_calculator(expression: str) -> str:
        return "400"

    start = time.perf_counter()
    try:
        chat = client.chats.create(
            model=target_model,
            config=types.GenerateContentConfig(
                tools=[dummy_calculator],
                temperature=0,
                max_output_tokens=100
            )
        )
        tool_resp = chat.send_message("What is 25 * 16? Use the dummy_calculator tool.")
        elapsed = time.perf_counter() - start
        text = tool_resp.text.strip() if tool_resp.text else "(tool executed)"
        print(f"        [STATUS] SUCCESS")
        print(f"        [AGENT RESPONSE] {text}")
        print(f"        [LATENCY] {elapsed:.2f} seconds")

    except Exception as error:
        elapsed = time.perf_counter() - start
        print(f"        [FAILED] Tool test failed after {elapsed:.2f} seconds")
        print(f"        [ERROR] {type(error).__name__}: {str(error)[:160]}")


print("=" * 65)
print("[DIAGNOSTIC] Gemini API Connectivity & Latency Benchmark")
print("=" * 65)
print(f"Configured Model in .env: {configured_model}")
print(f"HTTP Timeout:             {timeout_sec}s")

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=timeout_sec * 1000)
)

# Test configured model
run_diagnostic_on_model(configured_model, client)

# If configured model is not gemini-flash-lite-latest, benchmark gemini-flash-lite-latest as comparison
if configured_model != "gemini-flash-lite-latest":
    print("\n" + "-" * 65)
    print("Comparative Test: Recommended Fast Model ('gemini-flash-lite-latest')")
    print("-" * 65)
    fast_client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=15000)
    )
    run_diagnostic_on_model("gemini-flash-lite-latest", fast_client)

print("\n" + "=" * 65)
print("Diagnostic Complete.")
print("=" * 65)
