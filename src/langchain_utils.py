from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import VertexAIEmbeddings

def split_documents(text, chunk_size=1500, chunk_overlap=0):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(text)

# Embedding
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH =5

def create_embeddings(docs):
    embeddings = VertexAIEmbeddings(requests_per_minute=EMBEDDING_QPM,
                                    num_instances_per_batch=EMBEDDING_NUM_BATCH)
    return create_embeddings(docs)



def create_retriever(db):
    return db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
