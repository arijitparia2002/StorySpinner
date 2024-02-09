# src/langchain_utils.py
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from dotenv import load_dotenv
import os
import vertexai


def process_video(youtube_link):
    loader = YoutubeLoader.from_youtube_url(youtube_link, add_video_info=True)
    return loader.load()

def process_text(video_result):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
    return text_splitter.split_documents(video_result)

def initialize_langchain():
    # Load environment variables from .env file
    load_dotenv()

    # Get the Google Cloud Project ID from the environment variables
    GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

    llm = VertexAI(
        project=GOOGLE_CLOUD_PROJECT_ID,
        model_name="text-bison@001",
        max_output_tokens=256,
        temperature=0.1,
        top_p=0.8,
        top_k=40,
        verbose=True,
    )

    EMBEDDING_QPM = 100
    EMBEDDING_NUM_BATCH = 5
    embeddings = VertexAIEmbeddings(
        requests_per_minute=EMBEDDING_QPM,
        num_instances_per_batch=EMBEDDING_NUM_BATCH,
    )

    # Assuming docs is defined somewhere
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    return qa

def get_response(input_text, langchain_instance):
    video_subset = langchain_instance({"query": input_text})
    context = video_subset
    prompt = f"""
    Answer the following question in a detailed manner, using information from the text below.
    If the answer is not in the text, say 'HMM don't you try this out buddy' I don't know and do not generate your own response.

    Question:
    {input_text}
    Text:
    {context}

    Question:
    {input_text}

    Answer:
    """
    parameters = {
        "temperature": 0.1,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40
    }
    response = langchain_instance.predict(prompt, **parameters)
    return {"answer": response}
