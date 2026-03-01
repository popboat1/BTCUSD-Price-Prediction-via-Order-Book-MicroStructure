FROM python:3.10-slim

RUN apt-get update && apt-get install -y nodejs npm

WORKDIR /app

COPY src/requirements.txt .
RUN pip install -r requirements.txt

COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

COPY . .

EXPOSE 8000

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8001 & cd frontend && npm start -- -p 8000"]
