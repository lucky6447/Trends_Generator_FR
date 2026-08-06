import json
import re
import time

from ollama import chat
from config import MODEL


def clean_json(text):
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    match = re.search(r"\\{.*\\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    return text


def generate(prompt, retries=3):

    last_error = None

    for attempt in range(retries):

        try:
            response = chat(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
           options={
    "temperature": 0.25,
    "top_p": 0.8,
    "top_k": 40,
    "num_ctx": 8192,
    "num_predict": 4096
},
                format="json"
            )

            text = clean_json(response.message.content)
            return json.loads(text)

        except Exception as e:
            last_error = e
            print(f"[Retry {attempt+1}/{retries}] {e}")

            try:
                print(response.message.content[:1000])
            except:
                pass

            time.sleep(5)

    raise Exception(f"Ollama failed: {last_error}")
