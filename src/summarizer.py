import os
import asyncio
import logging
from typing import List, Union, Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import state

load_dotenv()


logger = logging.getLogger(__name__)

token_coefficient = state.token_coefficient
max_tokens = state.max_tokens
llm = ChatOpenAI(model=os.getenv("OPENAI_BASE_MODEL"), temperature=state.temperature)


def count_tokens(text: str, token_coefficient: int = token_coefficient) -> int:
    return len(text) // token_coefficient

def length_function(documents: List[Document]) -> int:
    length = sum(count_tokens(doc.page_content) for doc in documents)
    return length

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=max_tokens, chunk_overlap=32,
    separators=["\n\n", "\n", " ", ""],
    length_function=count_tokens
)

reduce_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ты — ассистент, который суммаризует текст на русском языке."),
        ("user", "Суммаризуй кратко следующий текст:\n\n{content}"),
    ]
)

def summary(text: str) -> str:
    logger.info('Start summary')
    try:
        if text.strip():
            docs = [Document(page_content=text)]
            summary = asyncio.run(_summary(docs))
        else:
            summary = ""
        logger.info('Success summary')
        return summary
    except Exception:
        logger.exception("Failed summary")
        content = text[:max_tokens] + text[-max_tokens:]
        try:
            prompt = reduce_prompt.invoke({"content": content})
            response = llm.invoke(prompt)
            return response.content
        except Exception:
            logger.exception("Failed exception summary")
            resume = " ".join([row for row in content.split("\n") if row.strip() and "SPEAKER_" not in row])
            return resume
    finally:
        logger.info('Finish summary')

async def reduce_chunks(input_chunks: Sequence[Union[Document, str]]) -> str:
    if input_chunks and isinstance(input_chunks[0], Document):
        content = "\n\n".join(doc.page_content for doc in input_chunks)
    else:
        content = "\n\n".join(input_chunks)
    prompt = reduce_prompt.invoke({"content": content})
    response = await llm.ainvoke(prompt)
    return response.content

def get_doc_lists(chunks: List[Document], length_function: (...), max_tokens: int = max_tokens) -> List[List[Document]]:
    try:
        return split_list_of_docs(chunks, length_function, max_tokens)
    except Exception:
        logger.exception("Failed get_doc_lists")
        text = " ".join([chunk.page_content for chunk in chunks])
        doc_lists = []
        for i in range(0, len(text), max_tokens):
            doc_list = Document(page_content=text[i:i+max_tokens])
            doc_lists.append([doc_list])
        return doc_lists

async def _summary(docs) -> str:
    chunks = text_splitter.split_documents(docs)
    doc_lists = get_doc_lists(chunks, length_function)
    summaries = await asyncio.gather(
        *[acollapse_docs(doc_list, reduce_chunks) for doc_list in doc_lists]
    )
    while length_function(summaries) > max_tokens:
        doc_lists = get_doc_lists(chunks, length_function)
        summaries = await asyncio.gather(
            *[acollapse_docs(doc_list, reduce_chunks) for doc_list in doc_lists]
        )
    summary = await reduce_chunks(summaries)
    return summary

if __name__ == "__main__":
    with open("tests/datasets/long_text.txt") as f:
        text = f.read()
    result = summary(text)
    print(result)
