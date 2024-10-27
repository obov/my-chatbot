from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader


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
        chain = (
            {
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)

    else:
        st.session_state["messages"] = []


def main_ui():

    st.set_page_config(
        page_title="챗문철",
        page_icon="🎉",
    )

    st.title("챗문철")

    st.markdown(
        """
    Welcome!
                
    Use this chatbot to ask questions to an AI about your files!

    Upload your files on the sidebar.
    """
    )

    with st.sidebar:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )


if __name__ == "__main__":
    load_dotenv()
    main_ui()
    main()
