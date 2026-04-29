from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import time
from pypdf import PdfReader
import gradio as gr

load_dotenv(override=True)

# -----------------------------
# Utilities
# -----------------------------
def push(text):
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": os.getenv("PUSHOVER_TOKEN"),
                "user": os.getenv("PUSHOVER_USER"),
                "message": text,
            },
            timeout=5
        )
    except Exception as e:
        print(f"Pushover error: {e}", flush=True)


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording unknown question: {question}")
    return {"recorded": "ok"}


# -----------------------------
# Tool schemas
# -----------------------------
record_user_details_json = {
    "name": "record_user_details",
    "description": "Store user contact details",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string"},
            "name": {"type": "string"},
            "notes": {"type": "string"},
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Store unanswered questions",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]


# -----------------------------
# Core Agent
# -----------------------------
class Me:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        self.model_fallback = "gemini-2.5-flash"
        self.model_primary = "gemini-3.1-flash-lite-preview"

        self.name = "Jonas Torres"

        self.linkedin = self._load_pdf("me/linkedin.pdf")
        self.summary = self._load_text("me/summary.txt")

    # -----------------------------
    # Loaders
    # -----------------------------
    def _load_pdf(self, path):
        text = ""
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text += content
        except Exception as e:
            print(f"PDF load error: {e}", flush=True)
        return text

    def _load_text(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Text load error: {e}", flush=True)
            return ""

    # -----------------------------
    # Prompt
    # -----------------------------
    def system_prompt(self):
        return f"""
You are acting as {self.name}. You are answering questions on {self.name}'s website.

Be professional and engaging.

If you don't know the answer:
→ use record_unknown_question tool.

If the user shows interest:
→ ask for their email and use record_user_details.

## Summary:
{self.summary}

## LinkedIn:
{self.linkedin}
"""

    # -----------------------------
    # Retry logic (RECURSIVE)
    # -----------------------------
    def call_model(self, messages, attempt=1, max_attempts=3):
        try:
            return self.client.chat.completions.create(
                model=self.model_primary,
                messages=messages,
                tools=tools,
            )

        except Exception as e:
            print(f"Attempt {attempt} failed: {e}", flush=True)

            if attempt >= max_attempts:
                print("Max retries reached. Using fallback model.", flush=True)
                return self.client.chat.completions.create(
                    model=self.model_fallback,
                    messages=messages,
                    tools=tools,
                )

            time.sleep(10)
            return self.call_model(messages, attempt + 1, max_attempts)

    # -----------------------------
    # History formatting
    # -----------------------------
    def format_history(self, history):
        formatted = []

        for h in history:
            if isinstance(h, dict):
                formatted.append(h)
            elif isinstance(h, (list, tuple)) and len(h) == 2:
                user, assistant = h
                if user:
                    formatted.append({"role": "user", "content": user})
                if assistant:
                    formatted.append({"role": "assistant", "content": assistant})

        return formatted

    # -----------------------------
    # Tool handling
    # -----------------------------
    def handle_tool_call(self, tool_calls):
        results = []

        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")

            print(f"Tool called: {name} with {args}", flush=True)

            tool_fn = globals().get(name)

            try:
                result = tool_fn(**args) if tool_fn else {"error": "tool not found"}
            except Exception as e:
                result = {"error": str(e)}

            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": call.id,
            })

        return results

    # -----------------------------
    # Chat function (FIXED)
    # -----------------------------
    def chat(self, message, history):
        messages = [
            {"role": "system", "content": self.system_prompt()}
        ] + self.format_history(history) + [
            {"role": "user", "content": message}
        ]

        max_loops = 5

        for _ in range(max_loops):
            response = self.call_model(messages)

            msg = response.choices[0].message

            # Tool calling loop
            if response.choices[0].finish_reason == "tool_calls":
                messages.append(msg)
                tool_results = self.handle_tool_call(msg.tool_calls)
                messages.extend(tool_results)
                continue

            # ✅ SAFE RETURN (fixes your crash)
            if msg.content:
                return msg.content

            if msg.tool_calls:
                return "Action completed."

            return "Sorry, I couldn't generate a response."

        return "Sorry, something went wrong (loop limit reached)."


# -----------------------------
# App entry point
# -----------------------------
if __name__ == "__main__":
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY")

    me = Me(api_key)

    gr.ChatInterface(
        fn=me.chat,
        title="Jonas Torres´ AI Twin",
        description="Ask me anything about Jonas´ professional life.",
    ).launch()