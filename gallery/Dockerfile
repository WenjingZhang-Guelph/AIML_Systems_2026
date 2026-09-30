FROM python:3.12-slim

WORKDIR /app

# Install Flask
RUN pip install --no-cache-dir flask gunicorn

# Copy app source
COPY app/ ./app/
COPY graphs_manifest.json .
COPY tags.json .

# Graph images are NOT baked into the image — they live in the sibling
# ../figures/ folder at the toolbox root and are bind-mounted at /figures
# by docker-compose.yml (see GRAPHS_DIR in app/server.py).

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60", "app.server:app"]
