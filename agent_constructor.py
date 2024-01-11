import os
import pickle
from pathlib import Path

from llama_index import (
    ServiceContext,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI
from llama_index.node_parser import SentenceSplitter
from llama_index.tools import QueryEngineTool, ToolMetadata
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=20)

logger = logging.getLogger(__name__)

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

async def build_agent_per_doc(nodes, file_base):
    print(file_base)

    vi_out_path = f"./data/replit_docs/{file_base}"
    summary_out_path = f"./data/replit_docs/{file_base}_summary.pkl"

    if not os.path.exists(vi_out_path):
        Path("./data/replit_docs/").mkdir(parents=True, exist_ok=True)
        # build vector index
        vector_index = VectorStoreIndex(nodes, service_context=service_context)
        vector_index.storage_context.persist(persist_dir=vi_out_path)

    else:
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vi_out_path),
            service_context=service_context,
        )

    # build summary index
    summary_index = SummaryIndex(nodes, service_context=service_context)

    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize"
    )

    # extract a summary
    if not os.path.exists(summary_out_path):
        Path(summary_out_path).parent.mkdir(parents=True, exist_ok=True)
        summary = str(
            await summary_query_engine.aquery(
                "Extract a concise 1-2 line summary of this document"
            )
        )
        with open(summary_out_path, "wb") as f:
          pickle.dump(summary, f)
    else:
        with open(summary_out_path, "rb") as f:
          summary = pickle.load(f)

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name=f"vector_tool_{file_base}",
                description="Useful for questions related to specific facts",
            ),
        ),
        QueryEngineTool(
            query_engine=summary_query_engine,
            metadata=ToolMetadata(
                name=f"summary_tool_{file_base}",
                description="Useful for summarization questions",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-4")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
        system_prompt=f"""\
You are a specialized agent designed to answer queries about the `{file_base}.html` part of the Replit docs.
You must ALWAYS use at least one of the tools provided when answering a question; do NOT rely on prior knowledge.\
""",
    )

    return agent, summary


async def build_agents(docs):
    node_parser = SentenceSplitter()

    # Build agents dictionary
    agents_dict = {}
    extra_info_dict = {}

    # # this is for the baseline

    for idx, doc in enumerate(docs):
      logger.info(f"Idx {idx}/{len(docs)}")
      
      nodes = node_parser.get_nodes_from_documents([doc])

      # ID will be base + parent
      file_path = Path(doc.metadata["path"])
      file_base = str(file_path.parent.stem) + "_" + str(file_path.stem)
      agent, summary = await build_agent_per_doc(nodes, file_base)

      agents_dict[file_base] = agent
      extra_info_dict[file_base] = {"summary": summary, "nodes": nodes}

    return agents_dict, extra_info_dict