# embeddings functions
from langchain.vectorstores import FAISS
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import time
from langchain_core.documents import Document


def create_embeddings(
        docs: list[Document], 
        chunk_size:int, 
        chunk_overlap:int,
        embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1", 
        ):
    """given a sequence of `Document` objects this fucntion will
    generate embeddings for it.
    
    ## argument
    :params docs (list[Document]) -> list of `list[Document]`
    :params chunk_size (int) -> chunk size in which documents are chunks
    :params chunk_overlap (int) -> the amount of token that will be overlapped between chunks
    :params embedding_model (str) -> the huggingspace model that will embed the documents 
    ## Return
    Tuple of embedding and chunks
    """
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
    )

    # Stage one: read all the docs, split them into chunks.
    st = time.time()
    print('Loading documents ...')

    chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
    et = time.time() - st
    print(f'Time taken: {et} seconds.')

    #Stage two: embed the docs.
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    print(f"create a total of {len(chunks)}")

    return embeddings,chunks