from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain.embeddings import VertexAIEmbeddings
from dotenv import load_dotenv
import os
import vertexai
from youtube_utils import get_youtube_link

def process_video(youtube_link):
  loader = YoutubeLoader.from_youtube_url(youtube_link, add_video_info=True)
  return loader.load()


def process_text(video_result):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
  docs = text_splitter.split_documents(video_result)
  return docs


def initialize_langchain(video_result):
  # Load environment variables from .env file
  load_dotenv()

  # Process text from video
  docs = process_text(video_result)

  EMBEDDING_QPM = 100
  EMBEDDING_NUM_BATCH = 5
  embeddings = VertexAIEmbeddings(
      requests_per_minute=EMBEDDING_QPM,
      num_instances_per_batch=EMBEDDING_NUM_BATCH,
  )

  db = Chroma.from_documents(docs, embeddings)
  retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

  # Return other components without creating llm here
  return retriever, embeddings


def get_response(input_text, langchain_instance):


  # Get Google Cloud Project ID from environment variables
  GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")

  # Initialize llm within get_response
  llm = VertexAI(
      project=GOOGLE_CLOUD_PROJECT_ID,
      model_name="text-bison@001",
      max_output_tokens=256,
      temperature=0.1,
      top_p=0.8,
      top_k=40,
      verbose=True,
  )

  video_subset = {"query": input_text}
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
  response = llm.predict(prompt, **parameters)
  return {"answer": response}

