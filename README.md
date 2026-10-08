# Asistente de documentos de la facultad (RAG)

Asistente que responde preguntas sobre documentos reales (reglamentos, plan de estudios, apuntes) y cita de donde saco cada respuesta.

Usa la API de Gemini (free tier).

## Estado
- [ ] Etapa 1: llamadas a la API de un LLM
- [ ] Etapa 2: RAG (chunking, embeddings, base vectorial)
- [ ] Etapa 3: evaluacion con set de preguntas
- [ ] Etapa 4: API + deploy
- [ ] Etapa 5: uso real y mejora continua

## Como correrlo
```bash
python -m venv .venv
.venv\Scripts\activate         # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # y pone tu API key de Google AI Studio
python listar_modelos.py       # para ver que modelos tenes disponibles
python etapa1_llamar_modelo.py
```

## Decisiones y aprendizajes
- Arranque con la API de Gemini por su free tier. Mas adelante: comparar con un modelo local (Ollama).
