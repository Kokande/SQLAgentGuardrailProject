FROM python:3.11-alpine

WORKDIR /SQLAgentProject

COPY src/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ .
WORKDIR src

EXPOSE 8000

CMD ["python", "main.py"]
