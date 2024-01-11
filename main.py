import asyncio
import logging
import os
import sys

import streamlit as st
from llama_index import (
  VectorStoreIndex,
)
from llama_index.agent import FnRetrieverOpenAIAgent
from llama_index.llms import OpenAI
from llama_index.objects import (
  ObjectIndex,
  SimpleToolNodeMapping,
)
from llama_index.tools import QueryEngineTool, ToolMetadata

from agent_constructor import build_agents
from custom_retriever import CustomObjectRetriever, CustomRetriever
from docs_loader import DocsLoader

logging.basicConfig(stream=sys.stdout, level=20)

logger = logging.getLogger(__name__)

DEBUG = True

if 'OPENAI_API_KEY' not in os.environ:
  sys.stderr.write("""
  You haven't set up your API key yet.
  
  If you don't have an API key yet, visit:
  
  https://platform.openai.com/signup

  1. Make an account or sign in
  2. Click "View API Keys" from the top right menu.
  3. Click "Create new secret key"

  Then, open the Secrets Tool and add OPENAI_API_KEY as a secret.
  """)
  exit(1)

st.set_page_config(page_title="Q&A with Replit docs",
                   page_icon="📖",
                   layout="centered",
                   initial_sidebar_state="auto",
                   menu_items=None)

st.title("Q&A with Replit docs 📖 🤓")

st.markdown("""

### 👨‍💻 Intro

We build a top-level agent that can orchestrate across the different document agents to answer any user query.

This `RetrieverOpenAIAgent` performs tool retrieval before tool use (unlike a default agent that tries to put all tools in the prompt).

Adding in reranking: we use Cohere reranker to better filter the candidate set of documents.

Adding in a query planning tool: we add an explicit query planning tool that’s dynamically created based on the set of retrieved tools.

""")


@st.cache_resource
def st_build_agents(_docs):
  return asyncio.run(build_agents(_docs))

@st.cache_resource
def build_base_engine(_extra_info_dict):
  all_nodes = [
    n for extra_info in _extra_info_dict.values() for n in extra_info["nodes"]
  ]

  base_index = VectorStoreIndex(all_nodes)
  return base_index.as_query_engine(similarity_top_k=4)

# @st.cache_resource
def build_tools(_agents_dict, _extra_info_dict):
  all_tools = []

  for file_base, agent in _agents_dict.items():
    summary = _extra_info_dict[file_base]["summary"]
    doc_tool = QueryEngineTool(
        query_engine=agent,
        metadata=ToolMetadata(
            name=f"tool_{file_base.replace('.', '_')}",
            description=summary,
        ),
    )
    all_tools.append(doc_tool)
  
  logger.info(all_tools[0].metadata)
  
  return all_tools


def build_custom_retriever(all_tools):
  tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)

  obj_index = ObjectIndex.from_objects(
    all_tools,
      tool_mapping,
      VectorStoreIndex,
  )
  vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

  return CustomObjectRetriever(
      CustomRetriever(vector_node_retriever), tool_mapping, all_tools)


def build_top_agent(_custom_obj_retriever, _llm):
  
  return FnRetrieverOpenAIAgent.from_retriever(
    _custom_obj_retriever,
      system_prompt=""" \
  You are an agent designed to answer queries about the documentation.
  Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
  """,
      llm=_llm,
      verbose=True,
  )

@st.cache_resource
def get_base_user_query(user_input):
  return base_query_engine.query(user_input)

@st.cache_resource
def get_top_agent_query(user_input):
  return top_agent.query(user_input)

@st.cache_resource
def index_docs(docs_user_input):
  r = DocsLoader(docs_user_input, docs_limit=5 if DEBUG else 500)
  return r.index_docs()

docs_user_input = st.sidebar.text_input("Docs Site", "docs.replit.com")

with st.sidebar:

  st.sidebar.markdown("Pull documentation using `wget` (recursive)")

  if st.button("Refresh Docs"):
    r = DocsLoader(docs_user_input)

    with st.spinner(f"Pulling docs from {docs_user_input}..."):
      r.load_docs()
      st.cache_resource.clear()

  st.markdown(
      "Clear cached indicies & agents and rebuild. This will take a few minutes"
  )

  if st.button("Clear Cache"):
    st.cache_resource.clear()

@st.cache_resource
def build_from_input(docs_user_input):

  llm = OpenAI(model_name="gpt-4-0613")
  docs = index_docs(docs_user_input)
  
  agents_dict, extra_info_dict = st_build_agents(docs)
  base_query_engine = build_base_engine(extra_info_dict)

  all_tools = build_tools(agents_dict, extra_info_dict)

  custom_obj_retriever = build_custom_retriever(all_tools)

  top_agent = build_top_agent(custom_obj_retriever, llm)
  return top_agent, base_query_engine

top_agent, base_query_engine = build_from_input(docs_user_input)

# Take input from the user
user_input = st.text_input("Enter Your Query", "")

query_type = st.radio("Query type", [
    "Base Engine",
    "Top Level",
])

# Display the input
if st.button("Submit"):
  st.write(f"Your Query: {user_input}")

  with st.spinner("Thinking..."):
    result = get_base_user_query(user_input) if query_type == 'Base Engine' else get_top_agent_query(user_input)

    # Display the results
    st.write(f"Answer: {str(result)}")
