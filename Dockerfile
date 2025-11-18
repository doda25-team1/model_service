#Build stage
FROM python:3.12-slim AS builder

WORKDIR /app

COPY requirements.txt .

RUN pip install --user --no-cache-dir -r requirements.txt

#Run stage
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

COPY src/ /app/src/
COPY smsspamcollection/ /app/smsspamcollection/

RUN mkdir -p /app/output

ENV PORT=8081

EXPOSE ${PORT}

CMD ["python", "src/serve_model.py"]
