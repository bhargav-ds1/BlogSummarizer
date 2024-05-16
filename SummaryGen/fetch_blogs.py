import requests
from bs4 import BeautifulSoup
from typing import List
import os
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from tqdm import tqdm


class FetchBlogs:
    """
        A class to fetch blog posts from a specified base URL and store them in a document store.
        Scrapes the blog posts website using beautifulsoup

        Attributes:
            docs (List[Document]): A list that stores the fetched documents as instances of the Document class.
            base_url (str): The base URL of the organization which is used to navigate to the main blog posts page.
    """

    def __init__(self) -> None:
        """
            Initializes the FetchBlogs class with an empty list for documents and a specified base URL.
        """
        self.docs = []
        self.base_url = 'https://jobleads.com'

    def _get_blog_text(self, link: str) -> str:
        """
            Fetches and extracts the text from a single blog post.

                Parameters:
                    link (str): The URL suffix for the blog post to fetch.

                Returns:
                    str: The text content of the blog post, stripped of extra space.
        """
        blog = requests.get(self.base_url + link)
        soup = BeautifulSoup(blog.content, "html.parser")
        blog_text = soup.find(['div'], {'class': 'article-blog__content'}).text
        # can also remove the explore more articles section at the end of each blog post
        return blog_text.strip()

    def fetch_blogs(self) -> List[Document]:
        """
            Fetches multiple blog posts from the base URL and parses details into Document objects.

                Returns:
                    List[Document]: A list of Document objects containing fetched blog content and metadata.
                Notes:
                    - Each blog post is contained in a single document and extra info such as the link,
                    category and posted_data of the blog post are stored for each document.
                    - The id of each document is set to the title of the blog. Helpful to easily fetch relevant document
                    based on the title.
        """
        page = requests.get(self.base_url + '/career-advice')
        soup = BeautifulSoup(page.content, "html.parser")
        tags = soup.find_all("a", {"class": 'article-list__item'})
        for tag in tqdm(tags):
            title = tag.find(['h1', 'h2', 'h3', 'h4'], {'class': "article-list__title"}).text
            link = tag.attrs['href']
            header = tag.find(['div'], {'class': "article-list__header"}).text.strip().split('\n')
            category = header[0].strip()
            posted_date = header[1].strip()
            # existing_summary = tag.find(['p'], {'class': 'article-list__summary'}).text
            blog_text = self._get_blog_text(link)
            self.docs.append(
                Document(text=blog_text, id_=title,
                         extra_info={'link': link, 'category': category, 'posted_date': posted_date}
                         )
            )
        return self.docs

    @staticmethod
    def save_blogs(documents: List[Document], dir_name: str = 'Data/DataStore') -> None:
        """
            Saves the list of Document objects to a specified directory using the SimpleDocumentStore.

                Parameters:
                    documents (List[Document]): The documents to save.
                    dir_name (str): The directory name where documents should be stored.
                Notes:
                    - This function can be updated to include different document store which can provide advanced
                    storing and retrieval capabilities.
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        docstore = SimpleDocumentStore()
        docstore.add_documents(documents)
        StorageContext.from_defaults(docstore=docstore).persist(persist_dir=dir_name)
