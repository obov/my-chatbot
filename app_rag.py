from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def main():

    vectorstore = FAISS.load_local(
        "vector_stores/faiss_all-in-challenge_winner",
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )
    retriever = vectorstore.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    llm = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                Context: {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    send_message("I'm ready! Ask away!", "ai", save=False)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    paint_history()

    message = st.chat_input("Ask anything about your file...")

    if message:
        send_message(message, "human")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        with st.chat_message("ai"):
            response = rag_chain.invoke(message)

    else:
        st.session_state["messages"] = []


def main_ui():

    st.set_page_config(
        page_title="챗문철",
        page_icon="🎉",
    )

    st.title("챗문철")


if __name__ == "__main__":
    load_dotenv()
    main_ui()
    main()
