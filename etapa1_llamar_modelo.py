"""
Etapa 1: llamar a un LLM por API (Gemini, free tier).

Que vas a ver en este script:
  1. Llamada basica (system prompt, tokens, temperatura)
  2. Streaming (la respuesta va saliendo de a poco)
  3. Salida estructurada (JSON con un formato que vos definis)
  4. Manejo de errores (incluido el 429 de limite de uso)

Ojo: el modelo NO sabe nada de tus documentos. Solo predice texto.
Eso es lo que resolvemos en la etapa 2 con RAG.
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

# --- Configuracion -------------------------------------------------------

load_dotenv()  # carga las variables del archivo .env

API_KEY = os.getenv("GEMINI_API_KEY")
MODELO = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY:
    raise SystemExit("Falta GEMINI_API_KEY en el .env")

# El cliente es el objeto que hace los requests HTTP a la API por vos
client = genai.Client(api_key=API_KEY)

SYSTEM = (
    "Sos un asistente que ayuda a estudiantes de Ingenieria en Sistemas. "
    "Responde en espanol rioplatense, claro y corto."
)


def mostrar_tokens(resp) -> None:
    """Los tokens son lo que se mide (y en planes pagos, lo que se cobra)."""
    uso = resp.usage_metadata
    print(
        f"[tokens] entrada={uso.prompt_token_count} "
        f"salida={uso.candidates_token_count} "
        f"razonamiento={uso.thoughts_token_count or 0}"
    )


# --- 1. Llamada basica ---------------------------------------------------

def preguntar(pregunta: str, temperatura: float = 0.3) -> str:
    """Manda una pregunta y devuelve el texto de la respuesta."""
    try:
        resp = client.models.generate_content(
            model=MODELO,
            contents=pregunta,                 # el mensaje del usuario
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,     # instrucciones generales
                temperature=temperatura,       # 0 = predecible, mas alto = variado
                max_output_tokens=2048,        # tope de tokens que puede generar
            ),
        )
    except errors.APIError as e:
        # e.code es el codigo HTTP: 400 pedido mal armado, 403 key invalida,
        # 429 pasaste el limite del free tier, 5xx problema del lado de Google
        return f"Error de la API ({e.code}): {e.message}"

    mostrar_tokens(resp)
    return resp.text or "(el modelo no devolvio texto)"


# --- 2. Streaming --------------------------------------------------------

def preguntar_streaming(pregunta: str) -> None:
    """Igual que arriba, pero imprime los pedazos a medida que llegan."""
    respuesta = client.models.generate_content_stream(
        model=MODELO,
        contents=pregunta,
        config=types.GenerateContentConfig(system_instruction=SYSTEM),
    )
    for trozo in respuesta:
        print(trozo.text or "", end="", flush=True)
    print()


# --- 3. Salida estructurada ----------------------------------------------

class Materia(BaseModel):
    """El formato exacto que queremos que devuelva el modelo."""
    materia: str
    tema: str
    dificultad: str  # facil / media / dificil


def extraer_materia(texto: str) -> Materia | None:
    """Le pedimos JSON con el formato de la clase Materia y lo parseamos."""
    resp = client.models.generate_content(
        model=MODELO,
        contents=texto,
        config=types.GenerateContentConfig(
            system_instruction=(
                "Extrae del texto la materia, el tema y la dificultad "
                "(facil, media o dificil)."
            ),
            temperature=0,                          # para extraer datos, determinista
            response_mime_type="application/json",  # le exigimos JSON
            response_schema=Materia,                # con exactamente estos campos
        ),
    )
    return resp.parsed  # ya viene convertido a un objeto Materia


# --- Main ----------------------------------------------------------------

if __name__ == "__main__":
    print(f"Usando modelo: {MODELO}\n")

    print("--- 1. Llamada basica ---")
    print(preguntar("Explicame en 3 lineas que es un deadlock."))

    print("\n--- 2. Streaming ---")
    preguntar_streaming("Dame un ejemplo cotidiano de cola vs pila.")

    print("\n--- 3. Salida estructurada ---")
    datos = extraer_materia(
        "Hoy tengo parcial de Redes II sobre subnetting y se me complica bastante."
    )
    print(datos)
    if datos:
        print("Solo la materia:", datos.materia)

    print("\n--- 4. Y la pregunta clave ---")
    print(preguntar("Cuantas materias optativas tiene el plan de Sistemas de la UCC?"))
    # Probablemente invente o diga que no sabe. Anotalo: ese es el problema que resuelve RAG.
