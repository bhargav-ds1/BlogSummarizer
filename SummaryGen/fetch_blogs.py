# import logging
# import sys
import requests
import pandas as pd
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
from bs4 import BeautifulSoup
import os, time
from llama_index.core.schema import Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext
from tqdm import tqdm

class FetchBlogs:
    def __init__(self):
        self.docs = []
        self.base_url = 'https://jobleads.com'
    def _get_blog_text(self, link: str) -> str:
        blog = requests.get(self.base_url + link)
        soup = BeautifulSoup(blog.content, "html.parser")
        blog_text = soup.find(['div'], {'class': 'article-blog__content'}).text
        # can also remove the explore more articles section
        return blog_text.strip()

    def get_titles(self, docs: list[Document] = None) -> list[str]:
        if docs:
            return [doc._id for doc in docs]
        elif os.path.exists(self.output_dir) and os.path.exists(self.output_dir + '/docstore.json'):
            blogs = SimpleDocumentStore().from_persist_dir(self.output_dir).docs
            return [doc.id_ for doc in blogs]
        else:
            blogs = self.fetch_blogs()
            return [doc._id for doc in blogs]

    def fetch_blogs(self):
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

    def save_blogs(self, documents, dir_name='Data/DataStore'):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        docstore = SimpleDocumentStore()
        docstore.add_documents(documents)
        StorageContext.from_defaults(docstore=docstore).persist(persist_dir=dir_name)

