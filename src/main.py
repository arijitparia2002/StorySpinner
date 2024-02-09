# src/main.py
from flask import Flask, render_template, request
from youtube_utils import get_youtube_link
from langchain_utils import process_video, process_text, initialize_langchain, get_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    youtube_link = request.form['youtube_link']
    video_result = process_video(youtube_link)
    text_result = process_text(video_result)
    langchain_instance = initialize_langchain()

    input_text = request.form['input_text']
    gr_interface_result = get_response(input_text, langchain_instance)

    return render_template('result.html', result=gr_interface_result)

if __name__ == "__main__":
    app.run(debug=True)
