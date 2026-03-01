FROM python:3.10-slim

RUN apt-get update && apt-get install -y nodejs npm

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "cd frontend && npm start & uvicorn src.api:app --host 0.0.0.0 --port 8000"]