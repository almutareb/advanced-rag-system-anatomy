# vectorization functions
from langchain.vectorstores import FAISS
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from create_embedding import create_embeddings
import time

def build_vector_store(
        docs: list, 
        db_path: str, 
        embedding_model: str, 
        new_db:bool=False, 
        chunk_size:int=500, 
        chunk_overlap:int=50,
        ):
    """

    """

    if db_path is None:
        FAISS_INDEX_PATH = "./vectorstore/py-faiss-multi-mpnet-500"
    else:
        FAISS_INDEX_PATH = db_path

    embeddings,chunks = create_embeddings(docs, embedding_model, chunk_size, chunk_overlap)

    #load chunks into vector store
    print(f'Loading chunks into faiss vector store ...')
    st = time.time()
    if new_db:
        db_faiss = FAISS.from_documents(chunks, embeddings)
    else:
        db_faiss = FAISS.add_documents(chunks, embeddings)
    db_faiss.save_local(FAISS_INDEX_PATH)
    et = time.time() - st
    print(f'Time taken: {et} seconds.')

    #print(f'Loading chunks into chroma vector store ...')
    #st = time.time()
    #persist_directory='./vectorstore/py-chroma-multi-mpnet-500'
    #db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    #et = time.time() - st
    #print(f'Time taken: {et} seconds.')
    result = f"built vectore store at {FAISS_INDEX_PATH}"
    return result