import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_document(file_path: str) -> list[Document]:
    path = Path(file_path)
    loaders = {".pdf": PyPDFLoader, ".txt": TextLoader}
    loader_cls = loaders.get(path.suffix.lower())
    if not loader_cls:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    logger.info(f"Loading document: {path.name}")
    return loader_cls(file_path).load()


def build_vector_store(file_paths: list[str]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    all_chunks = []
    for fp in file_paths:
        docs = load_document(fp)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        logger.info(f"{Path(fp).name} → {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
    )
    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
    )
    logger.info(f"Vector store created with {len(all_chunks)} total chunks.")
    return vector_store


def query(question: str, vector_store: Chroma, chat_history: list) -> dict:
    llm = ChatOpenAI(
        model_name=config.LLM_MODEL,
        temperature=0,
        openai_api_key=config.OPENAI_API_KEY,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": config.TOP_K, "score_threshold": config.SCORE_THRESHOLD},
    )

    # Convert chat history into LangChain message objects
    history_messages = []
    for entry in chat_history:
        history_messages.append(HumanMessage(content=entry["question"]))
        history_messages.append(AIMessage(content=entry["answer"]))

    # Prompt that includes chat history + retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions strictly based on the provided context.
If the answer is not in the context, say 'I couldn't find that in the provided documents.'
Do NOT make up information.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    source_docs = retriever.invoke(question)
    context = format_docs(source_docs)

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "history": history_messages,
        "question": question,
    })

    confidence = _confidence_score(question, vector_store)

    return {
        "answer": answer,
        "sources": source_docs,
        "confidence": confidence,
    }


def _confidence_score(question: str, vector_store: Chroma) -> float:
    results = vector_store.similarity_search_with_relevance_scores(
        question, k=config.TOP_K
    )
    if not results:
        return 0.0
    avg = sum(score for _, score in results) / len(results)
    return round(min(avg * 100, 100.0), 1)