FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable 2>/dev/null || uv sync --frozen --no-dev

COPY src ./src

CMD ["uv", "run", "python", "-m", "worker.entrypoints.worker"]
