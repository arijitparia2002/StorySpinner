import os
import vertexai
from dotenv import load_dotenv
from langchain_utils import sm_ask

load_dotenv()  # Load environment variables from .env file
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("GOOGLE_CLOUD_REGION")
vertexai.init(project=PROJECT_ID, location=REGION)

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/answer", methods=["POST"])
def answer():
    url = request.form["url"]
    question = request.form["question"]
    answer = sm_ask(url, question)  # Call sm_ask function
    print("The response is", answer)
    return render_template("result.html", answer=answer["answer"])

if __name__ == "__main__":

    app.run(debug=True)
