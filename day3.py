from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_community.document_loaders import WebBaseLoader


from pprint import pprint
from typing import List

from langchain_core.documents import Document
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph

def get_retriever():
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)

    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    )
    retriever = vectorstore.as_retriever()
    return retriever

class RelevanceResult(BaseModel):
    relevance: bool = Field(description="query relevance to the text chunk")

### State
class GraphState(TypedDict):
    question: str
    generation: str
    trial_tavily: int
    trial_generation: int
    web_search: str
    documents: List[str]

def n1_doc_retrieval(state):
    print("---RETRIEVE (n1_doc_retrieval) ---")
    question = state["question"]
    documents = retriever.invoke(question)
    print(question)
    print("documents retrieved: " + str(len(documents)))
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {doc.page_content[:20]} ...")
    return {"documents": documents, "question": question}

def e1_relevance_checker(state):
    print("--- (e1_relevance_checker) ---")
    parser = JsonOutputParser(pydantic_object=RelevanceResult)

    prompt = PromptTemplate(
        template="""
        You are an expert at evaluating information retrieval quality.
        Given a user query and a retrieved text chunk, determine if the chunk is relevant to the query.

        # Format: {format_instructions}
        # User Query: {query}
        # Retrieved Chunk: {text}
            """,
        partial_variables={"format_instructions": parser.get_format_instructions()},
        input_variables=["query", "text"],
    )

    question = state["question"]
    documents = state["documents"]
    trial_tavily = state.get("trial_tavily", 0)

    if trial_tavily > 1:
        print("FAILED")
        return "FAILED"

    chain = prompt | llm | parser
    result = chain.invoke({"query": question, "text": documents})

    if result['relevance']:
        print("TRUE")
        return "TRUE"
    else:
        print("FALSE")
        return "FALSE"


def e2_hallucination_checker(state):
    print("--- (e2_hallucination_checker) ---")

    system = """You are a grader assessing whether
        an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
        whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a
        single key 'score' and no preamble or explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "documents: {documents}\n\n answer: {generation} "),
        ]
    )

    documents = state["documents"]
    generation = state["generation"]

    trial_generation = state.get("trial_generation", 0)

    if trial_generation > 1:
        print("FAILED")
        return "FAILED"

    hallucination_grader = prompt | llm | JsonOutputParser()
    result = hallucination_grader.invoke({"documents": documents, "generation": generation})

    if result['score'] == "yes":
        print("TRUE")
        return "TRUE"

    print("FALSE")
    return "FALSE"  # Assuming no hallucination for simplicity


def n2_false_tavily(state):
    print("---WEB SEARCH---")
    question = state["question"]
    trial_tavily = state.get("trial_tavily", 0)

    # Web search
    response = tavily.search(question, num_results=3)
    documents = [Document(page_content=bs4.BeautifulSoup(item['content'], "html.parser").get_text()) for item in response['results']]

    if not documents:
        return {"documents": [], "question": question, "trial_tavily": trial_tavily + 1}

    return {"documents": documents, "question": question, "trial_tavily": trial_tavily + 1}


def n3_generate_answer(state):
    print("---GENERATE (n3_generate_answer) ---")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"generation": "No relevant documents found."}

    # Generate answer
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant. Given the following documents, answer the user's question.

        Question: {question}
        Documents: {documents}

        Answer:
        """,
        input_variables=["question", "documents"]
    )

    trial_generation = state.get("trial_generation", 0)

    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"question": question, "documents": "\n\n".join(doc.page_content for doc in documents)})

    return {"generation": generation, "trial_generation": trial_generation + 1}



load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_API_KEY"] =

llm = ChatOpenAI(model="gpt-4o-mini", temperature = 0)
tavily = TavilyClient(api_key=tavily_key)

retriever = get_retriever()

workflow = StateGraph(GraphState)
workflow.add_node("n1_doc_retrieval", n1_doc_retrieval)
workflow.add_node("n2_false_tavily", n2_false_tavily)
workflow.add_node("n3_generate_answer", n3_generate_answer)

workflow.set_entry_point("n1_doc_retrieval")
workflow.add_conditional_edges(
    "n1_doc_retrieval",
    e1_relevance_checker,
    {
        "TRUE": "n3_generate_answer",
        "FALSE": "n2_false_tavily",
        "FAILED": END
    },
)
workflow.add_conditional_edges(
    "n2_false_tavily",
    e1_relevance_checker,
    {
        "TRUE": "n3_generate_answer",
        "FALSE": "n2_false_tavily",
        "FAILED": END
    },
)
workflow.add_conditional_edges(
    "n3_generate_answer",
    e2_hallucination_checker,
    {
        "TRUE": END,
        "FALSE": "n3_generate_answer",
        "FAILED": END
    },
)


# workflow.add_edge("n2_false_tavily", "s2_relevance_checker")


# Compile
app = workflow.compile()

inputs = {"question": "what is prompt?"}
for output in app.stream(inputs):
    # print("do output", output)
    for key, value in output.items():
        pprint(f"Finished running: {key}:")

if value.get("trial_tavily", 0) > 1:
    pprint("failed: not relevant")
elif value.get("trial_generation", 0) > 1:
    pprint("failed: hallucination")
else :
    pprint(value["generation"])

