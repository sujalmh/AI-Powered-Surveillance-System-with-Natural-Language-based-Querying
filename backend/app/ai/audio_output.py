import base64
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
    messages=[
        {
            "role": "user",
            "content": """KEEP IT SHORT AND EXACT. Key features: real time object & person detection. smart search with natural langauge.  fast retrival with mongodb.
"""
        }
    ]
)

print(completion.choices[0])

wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
with open("dog.wav", "wb") as f:
    f.write(wav_bytes)