from typing import List
from langchain.document_loaders import TextLoader, YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

# Initialize the Vertex AI language model
model_name = "text-bison@001"
llm = VertexAI(model_name=model_name, max_output_tokens=256, temperature=0.1, top_p=0.8, top_k=40, verbose=True)

# Define the function to embed text using Vertex AI Language Models
def embed_text(texts: List[str], model_name: str) -> List[List[float]]:
    model = TextEmbeddingModel.from_pretrained(model_name)
    inputs = [TextEmbeddingInput(text, task="QUESTION_ANSWERING") for text in texts]
    embeddings = model.get_embeddings(inputs)
    return [embedding.values for embedding in embeddings]

def sm_ask(url: str, question: str, print_results: bool = True):
    try:
        if url.startswith("https://youtu.be/"):
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        else:
            loader = TextLoader(url)

        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=0)
        docs = text_splitter.split_documents(result)

        # Inspect the structure of the Document object
        print(docs[0])  # Print the first document
        # You may need to iterate through docs and print each document to understand its structure

        # Embed text content using Vertex AI Language Models
        texts = []
        for doc in docs:
            # Modify this part according to the actual structure of the Document object
            # If the text content is stored in a different attribute, use that attribute here
            text = getattr(doc, "text", None)
            if text:
                texts.append(text)

        embeddings = embed_text(texts, model_name)

        # Perform question answering using Langchain and Vertex AI
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=None, return_source_documents=True)
        video_subset = qa({"query": question})
        context = video_subset

        # Format the prompt for Vertex AI Language Model
        prompt = f"""
        Answer the following question in a detailed manner, using information from the text below. If the answer is not in the text, say 'HMM don't you try this out buddy' I don't know and do not generate your own response.

        Question:
        {question}

        Text:
        {context}

        Question:
        {question}

        Answer:
        """

        # Generate response using Vertex AI Language Model
        response = llm.predict(prompt)

        return {"answer": response}

    except Exception as e:
        print(f"Error processing URL: {e}")
        return {"answer": "Error: Could not process the URL."}
