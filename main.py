import os
import sys
from agent_constructor import build_agents
from docs_loader import DocsLoader
from custom_retriever import CustomObjectRetriever, CustomRetriever

from llama_index.agent import FnRetrieverOpenAIAgent
from llama_index.objects import (
    ObjectIndex,
    SimpleToolNodeMapping,
)

import asyncio
from llama_index.tools import QueryEngineTool, ToolMetadata
import logging

logging.basicConfig(stream=sys.stdout, level=20)

logger = logging.getLogger(__name__)

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

import streamlit as st
from llama_index import (
    VectorStoreIndex, )
from llama_index.llms import OpenAI

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
def build_agents(_docs):
  # what are agents?
  agents_dict, extra_info_dict = asyncio.run(build_agents(_docs))

  # define tool for each document agent
  all_tools = []

  for file_base, agent in agents_dict.items():
    summary = extra_info_dict[file_base]["summary"]
    doc_tool = QueryEngineTool(
        query_engine=agent,
        metadata=ToolMetadata(
            name=f"tool_{file_base}",
            description=summary,
        ),
    )
    all_tools.append(doc_tool)

  logger.info(all_tools[0].metadata)

  llm = OpenAI(model_name="gpt-4-0613")

  tool_mapping = SimpleToolNodeMapping.from_objects(all_tools)

  obj_index = ObjectIndex.from_objects(
      all_tools,
      tool_mapping,
      VectorStoreIndex,
  )
  vector_node_retriever = obj_index.as_node_retriever(similarity_top_k=10)

  custom_obj_retriever = CustomObjectRetriever(
      CustomRetriever(vector_node_retriever), tool_mapping, all_tools, llm=llm)

  top_agent = FnRetrieverOpenAIAgent.from_retriever(
      custom_obj_retriever,
      system_prompt=""" \
  You are an agent designed to answer queries about the documentation.
  Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
  """,
      llm=llm,
      verbose=True,
  )

  all_nodes = [
      n for extra_info in extra_info_dict.values() for n in extra_info["nodes"]
  ]

  base_index = VectorStoreIndex(all_nodes)
  base_query_engine = base_index.as_query_engine(similarity_top_k=4)

  return base_query_engine, top_agent


@st.cache_resource
def index_docs(_docs_user_input):
  r = DocsLoader(docs_user_input)
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

docs = index_docs(docs_user_input)
base_query_engine, top_agent = build_agents(docs)

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
    if query_type == 'Base Engine':
      result = base_query_engine.query(user_input)
    else:
      result = top_agent.query(user_input)

    # Display the results
    st.write(f"Answer: {str(result)}")
