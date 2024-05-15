import streamlit as st
import os
import sys
from typing import List, Tuple

st.set_page_config(  # Added favicon and title to the web app
    page_title="Blog Summariser",
    page_icon='favicon.ico',
    layout="wide",
    initial_sidebar_state="expanded",
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from SummaryGen.blog_summarizer import DocumentSummaryGenerator
from config import Config
from llama_index.core.base.response.schema import StreamingResponse

"""
This module is run using streamlit CLI command from the project root.
'Streamlit run Apps/Streamlit_app/app.py'
This renders the UI at http://localhost:8501/
"""

@st.cache_resource
def get_document_summarizer() -> Tuple[DocumentSummaryGenerator, List]:
    """
                This function connects to the Summary generator and provides an object along with a list of blog titles

                Parameters:

                Returns:
                    - DocumentSummaryGenerator (object), list of blog titles
                Notes:
                    - As, streamlit re-runs the whole application for every change. To avoid re-running the summary generator,
                this function result is cached by using the cache_resource decorator.


            """
    document_summarizer = DocumentSummaryGenerator(**Config['summarizer_args'], **Config['query_engine_args'])
    titles = document_summarizer.get_titles()
    return document_summarizer, titles


def makeStreamlitApp() -> None:
    """
                The UI of the streamlit app is built in this function. Which includes blog selection from a list of
                blogs and rendering the summary.
    """
    # Maintain a messages dict in session_state to avoid re-querying the summary of a blog every time it is selected
    if 'messages' not in st.session_state:
        st.session_state.messages = {}
    # Fetch the titles and the object summarizer object from a function which is cached.
    # (Avoids rebuilding the query engine, as streamlit tends to re-run the entire application)
    document_summarizer, titles = get_document_summarizer()
    st.title('Summary Generator')
    with st.sidebar:
        blog_id = st.selectbox('Select a blog to summarize',
                               options=titles, index=None, placeholder='Choose an option',
                               help='Select one of the titles of the blog to generate a summary of it.')
    st.header(str(blog_id) if blog_id else '', divider='rainbow')
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: aliceblue;
        border:none;
        color:black
    }
    div.stButton > button:hover {
        background-color: darkgray;
        border:none;
        color:black
        }
    </style>""", unsafe_allow_html=True)
    columns = st.columns([11, 1])
    with columns[1]:
        st.button('  ↻  ', on_click=lambda: st.session_state.messages.pop(blog_id))
    if blog_id in st.session_state.messages.keys():
        response = st.session_state.messages[blog_id]
    elif blog_id:
        response = document_summarizer.get_summary_response(doc_id=blog_id)
    else:
        response = ''
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
    """
    Entrypoint to the streamlit application.
    """
    makeStreamlitApp()
