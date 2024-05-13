import streamlit as st

st.set_page_config(  # Added favicon and title to the web app
    page_title="Blog Summariser",
    page_icon='favicon.ico',
    layout="wide",
    initial_sidebar_state="expanded",
)

from SummaryGen.blog_summarizer import DocumentSummaryGenerator
from config import Config
from llama_index.core.base.response.schema import StreamingResponse

@st.cache_resource
def get_document_summarizer():
    document_summarizer = DocumentSummaryGenerator(**Config['summarizer_args'], **Config['query_engine_args'])
    titles = document_summarizer.get_titles()
    return document_summarizer, titles


def makeStreamlitApp():
    # Maintain a messages dict in session_state to avoid re-querying the summary of a blog every time it is selected
    if 'messages' not in st.session_state:
        st.session_state.messages = {}
    # Fetch the titles and the object summarizer object from a function which is cached.
    # (Avoids rebuilding the query engine, as streamlit tends to re-run the entire application)
    document_summarizer, titles = get_document_summarizer()
    st.title('Summary Generator')
    with st.sidebar:
        blog_id = st.selectbox('Select a blog to summarize',
                               options=titles)
    st.header(str(blog_id), divider='rainbow')
    if blog_id in st.session_state.messages.keys():
        response = st.session_state.messages[blog_id]
    else:
        response = document_summarizer.get_summary_response(doc_id=blog_id)
    streaming = isinstance(response, StreamingResponse)
    if streaming:
        message_placeholder = st.empty()
        full_response = ""
        for res in response.response_gen:
            full_response += res
            message_placeholder.markdown(full_response + "▌ ")
        message_placeholder.markdown(full_response)
        st.session_state.messages[blog_id] = full_response
    else:
        st.markdown(response)
        st.session_state.messages[blog_id] = response


if __name__ == '__main__':
    makeStreamlitApp()
