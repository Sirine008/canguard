# canguard - CPU-only image (API + dashboard).
# The dataset and trained artifacts are NOT baked in: mount them at runtime.
#
#   docker build -t canguard .
#   docker run -p 8000:8000 ^
#     -v %cd%/data:/app/data -v %cd%/models:/app/models ^
#     -v %cd%/reports:/app/reports -v %cd%/knowledge_base:/app/knowledge_base ^
#     canguard
#
# Note: the Diagnose tab needs an Ollama server; point OLLAMA_HOST at one
# reachable from the container (e.g. http://host.docker.internal:11434).

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY frontend/ frontend/

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
