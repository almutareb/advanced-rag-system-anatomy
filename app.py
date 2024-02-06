import sys
import os

# from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader

# from bs4 import BeautifulSoup as Soup

# urls = ["https://langchain-doc.readthedocs.io/en/latest"]
# docs = []
# for url in urls:
#   loader = RecursiveUrlLoader(url=url, max_depth=5, extractor=lambda x: Soup(x, "html.parser").text)
#   docs.extend(loader.load())
# documents = loader.load()


# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import time

# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", "(?<=\. )", " ", ""],
#     chunk_size = 500,
#     chunk_overlap  = 50,
#     length_function = len,
# )

# # Stage one: read all the docs, split them into chunks.
# st = time.time()
# print('Loading documents ...')
# chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
# et = time.time() - st
# print(f'Time taken: {et} seconds.')


# from langchain.vectorstores import FAISS
# from langchain.vectorstores.utils import filter_complex_metadata
# import time
# from langchain.embeddings import HuggingFaceEmbeddings


# FAISS_INDEX_PATH = "./vectorstore/lc-faiss-multi-mpnet-500"


# #Stage two: embed the docs.
# # use all-mpnet-base-v2 sentence transformer to convert pieces of text in vectors to store them in the vector store
# model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# #model_kwargs = {"device": "cuda"}

# embeddings = HuggingFaceEmbeddings(
#     model_name=model_name,
# #    model_kwargs=model_kwargs
#     )
# print(f'Loading chunks into vector store ...')
# st = time.time()
# db = FAISS.from_documents(filter_complex_metadata(chunks), embeddings)
# db.save_local(FAISS_INDEX_PATH)
# et = time.time() - st
# print(f'Time taken: {et} seconds.')



from dotenv import load_dotenv
# load HF Token
config = load_dotenv(".env")
HUGGINGFACEHUB_API_TOKEN=os.getenv('HUGGINGFACEHUB_API_TOKEN')
# or use variable
#HUGGINGFACEHUB_API_TOKEN = ""


# HF libraries
from langchain.llms import HuggingFaceHub

model_id = HuggingFaceHub(repo_id="HuggingFaceH4/zephyr-7b-beta", model_kwargs={
    "temperature":0.1,
    "max_new_tokens":1024,
    "repetition_penalty":1.2,
    "streaming": True,
    "return_full_text":True
    })


import boto3
from botocore import UNSIGNED
from botocore.client import Config
import zipfile
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# access .env file

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

## Chroma DB
FAISS_INDEX_PATH='./vectorstore/lc-faiss-multi-mpnet-500-markdown'
VS_DESTINATION = FAISS_INDEX_PATH+".zip"
s3.download_file('rad-rag-demos', 'vectorstores/lc-faiss-multi-mpnet-500-markdown.zip', VS_DESTINATION)
#db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
#db.get()
with zipfile.ZipFile(VS_DESTINATION, 'r') as zip_ref:
    zip_ref.extractall('./vectorstore/')

model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
#model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
#    model_kwargs=model_kwargs
    )

db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

retriever = db.as_retriever(search_type = "mmr")#, search_kwargs={'k': 5, 'fetch_k': 25})

# retrieval chain
from langchain.chains import RetrievalQA
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


global qa
template = """
You are the friendly documentation buddy Arti, who helps the Human in using RAY, the open-source unified framework for scaling AI and Python applications.\
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question :
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)
memory = ConversationBufferMemory(memory_key="history", input_key="question")
qa = RetrievalQA.from_chain_type(llm=model_id, chain_type="stuff", retriever=retriever, verbose=True, return_source_documents=True, chain_type_kwargs={
    "verbose": True,
    "memory": memory,
    "prompt": prompt
}
    )


import gradio as gr
import random
import time

def add_text(history, text):
    history = history + [(text, None)]
    return history, ""

def bot(history):
    response = infer(history[-1][0], history)
    print(*memory)
    sources = [doc.metadata.get("source") for doc in response['source_documents']]
    src_list = '\n'.join(sources)
    print_this = response['result']+"\n\n\n Sources: \n\n\n"+src_list

    #history[-1][1] = ""
    #for character in response['result']: #print_this:
    #    history[-1][1] += character
    #    time.sleep(0.05)
    #    yield history
    history[-1][1] = print_this #response['result']
    return history

def infer(question, history):
    query =  question
    result = qa({"query": query, "history": history, "question": question})
    return result

css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""

title = """
<div style="text-align: center;max-width: 700px;">
    <h1>Chat with your Documentation</h1>
    <p style="text-align: center;">Chat with Documentation, <br />
    when everything is ready, you can start asking questions about the docu ;)</p>
</div>
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)
        chatbot = gr.Chatbot([], elem_id="chatbot")
        clear = gr.Button("Clear")
        with gr.Row():
            question = gr.Textbox(label="Question", placeholder="Type your question and hit Enter ")
    question.submit(add_text, [chatbot, question], [chatbot, question], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=False)