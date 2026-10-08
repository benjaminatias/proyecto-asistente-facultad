"""
Muestra que modelos de Gemini podes usar con tu key.
Sirve para elegir el valor de GEMINI_MODEL en el .env.
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for modelo in client.models.list():
    # Nos quedamos solo con los que sirven para generar texto
    if "generateContent" in (modelo.supported_actions or []):
        print(modelo.name)
