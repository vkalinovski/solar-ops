FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY solarflare_app ./solarflare_app
COPY backend ./backend
COPY scripts ./scripts
RUN pip install --no-cache-dir .

EXPOSE 8000
CMD ["python", "scripts/serve.py"]
