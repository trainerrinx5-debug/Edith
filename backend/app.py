"""
E.D.I.T.H. backend — Phase 1 (Foundation)

A tiny Flask server whose only job is to:
  1. Receive chat messages from the frontend
  2. Forward them to Groq (fast, free-tier AI inference)
  3. Send the reply back

Why a backend at all, instead of calling Groq straight from the browser?
Because your Groq API key would be visible to anyone who opens dev tools
on the website. This server keeps the key secret (stored in Replit's
Secrets tab, never in code) and is the only thing that talks to Groq.

Run this on Replit: https://replit.com -> Create Repl -> Python.
"""

import os
import sys
import traceback

print("=== E.D.I.T.H. backend starting ===", flush=True)

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    from groq import Groq
    print("=== imports OK ===", flush=True)
except Exception:
    print("=== IMPORT FAILED ===", flush=True)
    traceback.print_exc()
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # allows your frontend (hosted anywhere) to call this backend

GROQ_KEY = os.environ.get("GROQ_API_KEY")
print(f"=== GROQ_API_KEY present: {bool(GROQ_KEY)} ===", flush=True)

try:
    client = Groq(api_key=GROQ_KEY)
    print("=== Groq client created OK ===", flush=True)
except Exception:
    print("=== GROQ CLIENT CREATION FAILED ===", flush=True)
    traceback.print_exc()
    sys.exit(1)

# Fast, free-tier Groq models. 8b-instant is faster; 70b-versatile is smarter.
DEFAULT_MODEL = "llama-3.3-70b-versatile"

EDITH_SYSTEM_PROMPT = """You are E.D.I.T.H., a personal AI assistant.
You are calm, precise, and quietly capable — like a highly competent
right hand, not a chatty chatbot. Keep answers concise unless asked
to elaborate. Never claim to control real device hardware unless the
frontend explicitly tells you a tool result. If asked to do something
you (as a language model) cannot actually do, say so plainly."""


@app.route("/")
def health():
    # Visiting the backend URL directly should show this, so you know
    # it's alive when debugging from the tablet browser.
    return jsonify({"status": "E.D.I.T.H. backend is running"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True) or {}
    messages = data.get("messages", [])
    tools = data.get("tools")  # tool schemas sent by the frontend's toolManager.js

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # Prepend the personality/system prompt every time
    full_messages = [{"role": "system", "content": EDITH_SYSTEM_PROMPT}] + messages

    try:
        kwargs = dict(
            model=data.get("model", DEFAULT_MODEL),
            messages=full_messages,
            temperature=0.7,
            max_tokens=1024,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        completion = client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message

        # If the model wants to call a tool, hand the request back to the
        # frontend — the backend never executes device actions itself.
        if msg.tool_calls:
            return jsonify({
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

        return jsonify({"reply": msg.content})
    except Exception as e:
        # Never let the server crash on a bad API response — the frontend
        # needs a clean error it can show the user.
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"=== Starting Flask on 0.0.0.0:{port} ===", flush=True)
    app.run(host="0.0.0.0", port=port)

