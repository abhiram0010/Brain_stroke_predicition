import os
import json
import base64
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env file
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_image(image_bytes):
    """Send image to Groq API to check if it's a Brain CT scan."""
    try:
        # Convert the image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Prepare the message for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is this a Brain CT image? Respond with a JSON object containing a 'is_brain_CT' boolean field."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]

        # Send request to Groq API
        response = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=50
        )

        # Parse response
        response_data = json.loads(response.choices[0].message.content)
        return response_data.get("is_brain_CT", False)

    except Exception as e:
        print(f"❌ Error in API call: {e}")
        return False  # Return False if API fails

def main(image_bytes):
    """Main function to check if uploaded image is a Brain CT scan."""
    return analyze_image(image_bytes)
