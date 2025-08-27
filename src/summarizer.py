import os
import asyncio
from typing import List, Union, Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.reduce import acollapse_docs, split_list_of_docs
from langchain_text_splitters import TokenTextSplitter

load_dotenv()

llm = ChatOpenAI(model=os.getenv("OPENAI_BASE_MODEL"), temperature=0)

token_max = 1024
text_splitter = TokenTextSplitter(
    chunk_size=token_max, chunk_overlap=32
)

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
    resp = await llm.ainvoke(prompt)
    return resp.content

async def main(docs) -> str:
    chunks = text_splitter.split_documents(docs)
    # 1-й уровень: параллельное схлопывание чанков
    doc_lists = split_list_of_docs(chunks, length_function, token_max)
    print(len(doc_lists))
    summaries = await asyncio.gather(
        *[acollapse_docs(doc_list, reduce_chunks) for doc_list in doc_lists]
    )
    print([llm.get_num_tokens(summary.page_content) for summary in summaries])
    # Иерархическое слияние без синхронных блокировок
    while length_function(summaries) > token_max:
        doc_lists = split_list_of_docs(summaries, length_function, token_max)
        print(len(doc_lists))
        summaries = await asyncio.gather(
            *[acollapse_docs(doc_list, reduce_chunks) for doc_list in doc_lists]
        )
        print([llm.get_num_tokens(summary.page_content) for summary in summaries])
    summary = await reduce_chunks(summaries)
    print(llm.get_num_tokens(summary))
    return summary

if __name__ == "__main__":
    with open("tests/datasets/long_text.txt") as f:
        content = f.read()
    docs = [Document(page_content=content)]
    result = asyncio.run(main(docs))
    assert isinstance(result, str)
    print(result)
