# Dockerfile.app
# Long-running Streamlit service.

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

EXPOSE 8501

ENTRYPOINT ["uv", "run", "streamlit", "run", "src/streamlit_app.py", \
    "--server.address=0.0.0.0", "--server.port=8501"]