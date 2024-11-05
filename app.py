from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationChain
from langchain.callbacks.base import BaseCallbackHandler
from store import get_session_history


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


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
        "vector_stores/faiss_doorfe",
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=1,
        streaming=True,
        callbacks=[
            ChatCallbackHandler(),
        ],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                
                Context: {context}
                """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.get_relevant_documents(x["question"])
            )
        )
        | prompt
        | llm
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션 기록을 가져오는 함수
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",  # 기록 메시지의 키
    )

    send_message("안녕하세요!", "ai", save=False)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    paint_history()

    message = st.chat_input("질문을 입력해주세요.")

    if message:
        send_message(message, "human")

        with st.chat_message("ai"):
            response = chain_with_history.invoke(
                {"question": message},
                config={"configurable": {"session_id": "abc123"}},
            )
            # response = chain.invoke(message)
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


if __name__ == "__main__":
    load_dotenv()
    main_ui()
    main()
