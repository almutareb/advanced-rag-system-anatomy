# documents loader function
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from validators import url as url_validator
from langchain_core.documents import Document

def load_docs_from_urls(
        urls: list = ["https://docs.python.org/3/"], 
        max_depth: int = 5,
        ) -> list[Document]:
    """
    Load documents from a list of URLs.

    ## Args:
        urls (list, optional): A list of URLs to load documents from. Defaults to ["https://docs.python.org/3/"].
        max_depth (int, optional): Maximum depth to recursively load documents from each URL. Defaults to 5.

    ## Returns:
        list: A list of documents loaded from the given URLs.
        
    ## Raises:
        ValueError: If any URL in the provided list is invalid.
    """

    docs = []
    for url in urls:
        if not url_validator(url):
            raise ValueError(f"Invalid URL: {url}")
        loader = RecursiveUrlLoader(url=url, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser").text)
        docs.extend(loader.load())
    print(f"loaded {len(docs)} pages")
    return docs