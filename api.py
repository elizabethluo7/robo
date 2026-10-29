
from google import genai
from google.genai import types
import os

# Read your API key from the environment
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Set GEMINI_API_KEY in your terminal first.")

client = genai.Client(api_key=api_key)
question = "Describe what you see in this image."
image_path = "trial_image.png"

with open(image_path, "rb") as f:
    image_bytes = f.read()

# Send the image and question together
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[
        types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/png"
        ),
        question
    ]
)

print(response.text)
