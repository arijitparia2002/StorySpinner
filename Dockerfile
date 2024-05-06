FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]