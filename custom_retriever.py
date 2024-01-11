from llama_index.objects import ObjectRetriever
from llama_index.retrievers import BaseRetriever
from llama_index.postprocessor import CohereRerank
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index import ServiceContext
from llama_index.llms import OpenAI


# define a custom retriever with reranking
class CustomRetriever(BaseRetriever):

  def __init__(self, vector_retriever, postprocessor=None):
    self._vector_retriever = vector_retriever
    self._postprocessor = postprocessor or CohereRerank(top_n=5)
    super().__init__()

  def _retrieve(self, query_bundle):
    retrieved_nodes = self._vector_retriever.retrieve(query_bundle)
    filtered_nodes = self._postprocessor.postprocess_nodes(
        retrieved_nodes, query_bundle=query_bundle)

    return filtered_nodes


# define a custom object retriever that adds in a query planning tool
class CustomObjectRetriever(ObjectRetriever):

  def __init__(self, retriever, object_node_mapping, all_tools, llm=None):
    self._retriever = retriever
    self._object_node_mapping = object_node_mapping
    self._llm = llm or OpenAI("gpt-4-0613")

  def retrieve(self, query_bundle):
    nodes = self._retriever.retrieve(query_bundle)
    tools = [self._object_node_mapping.from_node(n.node) for n in nodes]

    sub_question_sc = ServiceContext.from_defaults(llm=self._llm)
    sub_question_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=tools, service_context=sub_question_sc)
    sub_question_description = f"""\
Useful for any queries that involve comparing multiple documents. ALWAYS use this tool for comparison queries - make sure to call this \
tool with the original query. Do NOT use the other tools for any queries involving multiple documents.
"""
    sub_question_tool = QueryEngineTool(
        query_engine=sub_question_engine,
        metadata=ToolMetadata(name="compare_tool",
                              description=sub_question_description),
    )

    return tools + [sub_question_tool]
