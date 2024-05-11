import streamlit as st
st.set_page_config(  # Added favicon and title to the web app
    page_title="Blog Summariser",
    page_icon='favicon.ico',
    layout="wide",
    initial_sidebar_state="expanded",
)

from SummaryGen.blog_summarizer import DocumentSummaryGenerator
from config import Config


# from Guardrails.NemoGuardrails import NemoGuardrails
# ngr = NemoGuardrails()

@st.cache_resource
def get_document_summarizer():
    document_summarizer = DocumentSummaryGenerator(**Config['summarizer_args'],
                                                   )
    titles = document_summarizer.get_titles()
    return document_summarizer, titles


def makeStreamlitApp():
    document_summarizer, titles = get_document_summarizer()
    st.title('Summary Generator')
    with st.sidebar:
        blog_id = st.selectbox('Select a blog to summarize',
                               options=titles)
    st.header(str(blog_id), divider='rainbow')
    summary = document_summarizer.get_summary(doc_id=blog_id)
    st.write(str(summary))


if __name__ == '__main__':
    makeStreamlitApp()
