"""LangChain retrieval chain for the internal docs Q&A assistant.

Built with LCEL (LangChain Expression Language): small "Runnables" joined
with ``|`` into a pipeline. Every step is traced in LangSmith when
LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set.

    question -> retriever -> prompt -> chat model -> string parser -> answer
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

DOCS = [
    Document(page_content="Deploys to production are frozen from 5 PM Friday until 9 AM Monday.", metadata={"id": "deploy-freeze"}),
    Document(page_content="Every production deploy must have one approving review from a code owner.", metadata={"id": "deploy-review"}),
    Document(page_content="On-call engineers must acknowledge a page within 15 minutes.", metadata={"id": "oncall-ack"}),
    Document(page_content="Incident postmortems are due within 5 business days of resolution.", metadata={"id": "postmortem"}),
]

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question using only this context:\n\n{context}"),
        ("human", "{question}"),
    ]
)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(question: str, k: int = 2) -> list[Document]:
    q = _tokens(question)
    return sorted(DOCS, key=lambda d: -len(q & _tokens(d.page_content)))[:k]


def build_chain(llm: BaseChatModel):
    """Return a Runnable mapping {"question": str} -> {"answer": str, "contexts": list[str]}."""
    retriever = RunnableLambda(lambda x: [d.page_content for d in retrieve(x["question"])])
    answer = (
        RunnablePassthrough.assign(context=lambda x: "\n".join(x["contexts"]))
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return RunnablePassthrough.assign(contexts=retriever) | RunnablePassthrough.assign(answer=answer)
