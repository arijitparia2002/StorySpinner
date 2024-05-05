import vertexai
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
REGION = os.getenv("GOOGLE_CLOUD_REGION")
vertexai.init(project=PROJECT_ID, location=REGION)

# Initialize VertexAI object with required arguments
MODEL = "text-embedding-preview-0409"
TASK = "RETRIEVAL_DOCUMENT"
OUTPUT_DIMENSIONALITY = 256

def sm_ask(question, video_url):
    video_subset = load_youtube_video(video_url)
    docs = split_documents(video_subset)
    embeddings = create_embeddings(docs)
    context = " ".join(embeddings)
    prompt = f"""
        Answer the following question in a detailed manner, using information from the text below. If the answer is not in the text, say I don't know and do not generate your own response.

        Question:
        {question}
        Text:
        {context}

        Question:
        {question}

        Answer:
        """
    response = embed_text(MODEL, TASK, prompt, output_dimensionality=OUTPUT_DIMENSIONALITY)
    return {"answer": response}

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/answer", methods=["POST"])
def answer():
    url = request.form["url"]
    question = request.form["question"]
    answer = sm_ask(question, url)
    return render_template("result.html", answer=answer["answer"])

if __name__ == "__main__":
    app.run(debug=True)
