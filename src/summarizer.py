import os
import asyncio
import logging
from typing import List, Union, Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langchain_text_splitters import TokenTextSplitter

load_dotenv()

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model=os.getenv("OPENAI_BASE_MODEL"), temperature=0)

token_max = 1024
text_splitter = TokenTextSplitter(
    chunk_size=token_max, chunk_overlap=32
)


def summary(text: str) -> str:
    if text.strip():
        docs = [Document(page_content=text)]
        summary = asyncio.run(_summary(docs))
    else:
        summary = ""
    logger.info(f"{summary=}")
    return summary

    

def length_function(documents: List[Document]) -> int:
    length = sum(llm.get_num_tokens(doc.page_content) for doc in documents)
    return length

reduce_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Ты — ассистент, который суммаризует текст на русском языке."),
        ("user", "Суммаризуй кратко следующий текст:\n\n{content}"),
    ]
)

async def reduce_chunks(input_chunks: Sequence[Union[Document, str]]) -> str:
    if input_chunks and isinstance(input_chunks[0], Document):
        content = "\n\n".join(d.page_content for d in input_chunks)
    else:
        content = "\n\n".join(input_chunks)
    prompt = reduce_prompt.invoke({"content": content})
    response = await llm.ainvoke(prompt)
    return response.content

async def _summary(docs) -> str:
    chunks = text_splitter.split_documents(docs)
    doc_lists = split_list_of_docs(chunks, length_function, token_max)
    logger.info(f"{len(doc_lists)=}")
    summaries = await asyncio.gather(
        *[acollapse_docs(doc_list, reduce_chunks) for doc_list in doc_lists]
    )
    while length_function(summaries) > token_max:
        doc_lists = split_list_of_docs(summaries, length_function, token_max)
        logger.info(f"{len(doc_lists)=}")
        summaries = await asyncio.gather(
            *[acollapse_docs(doc_list, reduce_chunks) for doc_list in doc_lists]
        )
    summary = await reduce_chunks(summaries)
    logger.info(f"{llm.get_num_tokens(summary)=}")
    return summary
