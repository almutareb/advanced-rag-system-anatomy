# preprocessed vectorstore retrieval
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import zipfile
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# access .env file

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
#model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
#    model_kwargs=model_kwargs
    )

## FAISS
FAISS_INDEX_PATH='./vectorstore/lc-faiss-multi-mpnet-500-markdown'
VS_DESTINATION = FAISS_INDEX_PATH+".zip"
s3.download_file('rad-rag-demos', 'vectorstores/lc-faiss-multi-mpnet-500-markdown.zip', VS_DESTINATION)
with zipfile.ZipFile(VS_DESTINATION, 'r') as zip_ref:
    zip_ref.extractall('./vectorstore/')
faissdb = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

## Chroma DB
chroma_directory="./vectorstore/lc-chroma-multi-mpnet-500-markdown"
VS_DESTINATION = chroma_directory+".zip"
s3.download_file('rad-rag-demos', 'vectorstores/lc-chroma-multi-mpnet-500-markdown.zip', VS_DESTINATION)
with zipfile.ZipFile(VS_DESTINATION, 'r') as zip_ref:
    zip_ref.extractall('./vectorstore/')
chromadb = Chroma(persist_directory=chroma_directory, embedding_function=embeddings)
chromadb.get()