# retriever and qa_chain function

# HF libraries
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
# vectorestore
from langchain.vectorstores import FAISS
# retrieval chain
from langchain.chains import RetrievalQA
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


def get_db_retriever(vector_db:str=None):
    model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
    embeddings = HuggingFaceHubEmbeddings(repo_id=model_name)

    #db = Chroma(persist_directory="./vectorstore/lc-chroma-multi-mpnet-500", embedding_function=embeddings)
    #db.get()
    if not vector_db:
        FAISS_INDEX_PATH='./vectorstore/py-faiss-multi-mpnet-500'
    else:
        FAISS_INDEX_PATH=vector_db
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

    retriever = db.as_retriever()

    return retriever