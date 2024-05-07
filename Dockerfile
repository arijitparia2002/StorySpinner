FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y build-essential && \
    pip install --upgrade pip

# Create a virtual environment
RUN python -m venv /venv

# Use the virtual environment
ENV PATH="/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]