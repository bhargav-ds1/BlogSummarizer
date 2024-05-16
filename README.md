# Blog Post Summarizer for JobLeads

## Overview

This project automates the summarization of blog posts from JobLeads website. It fetches blog posts, stores them in a
Simple Document Store, and uses an LLM of choice to generate summaries. This implementation depends on the Llama index
framework.

## Features

- **Automated Content Fetching**: Retrieves blog posts directly from JobLeads
  website (https://www.jobleads.com/career-advice).
- **Document Store**: Manages blog data efficiently by storing them as Document objects locally.
- **LLM-based Summarization**: Uses LLM models to create summaries. Tested with (meta-llama/Llama-2-7b-chat-hf,
  mistralai/Mixtral-8x7B-Instruct-v0.1) models downloaded from huggingface and LLM inference API provided by
  Together-AI.
- **Custom Retrieval**: The retrieval for the summarization task is straight-forward, so a custom retriever which
  retrieves the blog content based on the title of the blog is implemented. Optimally skipping the need to embed the
  data and retrieval based on embeddings.
- **Summarization strategy**: Two strategies namely, simple_summarize and tree_summarize are tested. While
  simple_summarize uses one single LLM call and truncates the data which exceeds the context length of the LLM.
  tree_summarize strategy summarizes the chunks/nodes recursively forming a tree structured approach. It combines chunks
  such that it can fill the context
  length of the LLM for each LLM call, obtains summaries and summarizes them recursively until a single summary is
  generated.
- **Testing/Evaluation**: To evaluate the performance of the LLM in creating the summaries, a framework provided by
  confident-ai known as Deepeval is utilized. The performance is tested/evaluated by using relevant metrics such as
  AnswerRelevancyMetric, SummarizationMetric, FaithfulnessMetric, HallucinationMetric and ToxicityMetric.
- **Observability**: Gathering the traces of an LLM application is important to monitor the performance and usage of the
  application. It helps in ensuring compliance with ethical standards, security and integrity along with support in
  optimizing and improving the model performance. Arize-phoenix as the observability framework is used, it provides an
  easy integration with llama index and provides a nice UI to visualize the traces locally.
- **Streamlit UI**: A basic UI which is built using streamlit is used as an entry point to the application. While other
  apps such as a Web API or a command line tool can also be built for this purpose. The UI provides functionality to
  select from a list of blog titles to generate a summary. Optionally also to regenerate the summary.

## Project Boundaries

- **Purpose**: The project aims only to summarize the blogs, so the application restricts its use only for summarization
  purposes. (i.e user input is taken only in the form of selection.). While, it is tempting to have additional
  functionality such as chatbot which can be used by the user to query the blog content it is treated as out of scope to
  avoid introducing extra complications.
- **Manually initiate blog Re-Fetching**: The blogs are fetched automatically, but if a new blog is added to the
  website, then the blogs are to be refetched. This is done by setting a boolean variable in the config file. In future,
  an event-driven approach which re-fetches when new blogs are added would be great to maintain timely data.
- **LLM providers**: There are a lot of providers of LLMs which provide inference APIs to access and use LLMs. Only some
  of the popular one's are considered in this project currently. The decision is biased to reduce or avoid incurring any
  charges.

## Prerequisites

- Python 3.8+
- Pip
- (LLM inference) Registered account to access LLMs
    - HuggingFace (Can be used to download opensource LLM models and use it for local inference. May result in slow
      response generation depending on the system specifications.). If you want to use an LLM by inferring it locally,
      then the model_download_script.py in the SummaryGen package can be used to download and test the models from
      huggingface. (Used during development and testing)
    - Together-AI (Provides an LLM inference API for some of the open source models. Can be used for free using the free
      initial credit.) (Used during development and testing)
    - OpenAI (Most popular proprietary LLM inference API, provides access to GPT models, requires OpenAI account with
      API access.) (Not used)
- (optional) (Tests/evaluations) Registered account Confident.ai (deepeval)
    - https://app.confident-ai.com/auth/signup
    - Provides a UI to visualize the evaluation results. If not the evaluation results are also visible in the
      command line.

## Installation

**Clone or navigate to the project**:

If you already have the code as a zip file then skip cloning, unzip the contents and navigate into the BlogSummarizer
directory.

```bash
git clone https://github.com/bhargav-ds1/BlogSummarizer.git
cd BlogSummarizer
```

**Create a virtual env and install dependencies**:

Create a virtual env using pip or anaconda.
Install the project dependencies

```bash
pip install -r requirements.txt
```

**Create required accounts if not already created (mostly free trails - incurs no charges)**:

- TogetherAI (https://api.together.ai/) (Favoured)
- HuggingFace (https://huggingface.co/)
- OpenAI (https://platform.openai.com/)

**Fill in API keys and other required config if any**:
Fill the API key for accessing the LLMs in the .envfile.

## Usage

**Summarization using the streamlit UI**:

```bash
streamlit run Apps/Streamlit_app/app.py
# or alternatively using the make command
make start-app
```

Alternatively, you can use the dockerized implementation by running

```bash
docker compose up
# or alternatively using the make command
make start-docker
```

The web UI can be accessed at (http://localhost:8501/)
The phoenix observability UI can be accessed at (http://localhost:6006/)

**Test summarization using deepeval**:

```bash
deepeval test run Tests/test_blog_summarizer.py
# or alternatively using the make command
make start-test
```

You can run `deepeval login` before running the test or uncomment the line containing the command in the make file if
you would like to use the web UI to visualize the
tests and have obtained relevant API key.

## Configuration

Modify the settings via the config.py file.

The configuration from the config.py file is used by the Apps (currently streamlit app) to provide configuration for the
DocumentSummaryGenerator class. It contains arguments related to sourcing and inferring the LLM, data related arguments
along with arguments for the query_engine (aka, the summarizer which uses the LLM to generate summaries based on a
strategy).

## Examples
**Example streamlit UI**

**Example Phoenix observability UI**

**Example deepeval evaluation output**

**Example deepeval evaluation UI**

## License

This project is licensed under the Apache License - see the LICENSE.md file for details.

## Acknowledgments

- JobLeads for blog access.


