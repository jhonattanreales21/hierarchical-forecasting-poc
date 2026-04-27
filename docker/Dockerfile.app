FROM python:3.12-slim
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY . .
RUN uv sync --package hdf_app
CMD ["uv", "run", "--package", "hdf_app", "streamlit", "run", "app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
