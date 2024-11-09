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
from UserIdManager import UserIdManager

user_id_manager = UserIdManager()


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
                "EXAMPLE CONVERSATION STARTS",
            ),
            (
                "system",
                """
                지금의 대화는 예시 대화로 보행자와의 사고시에 어떻게 대답하면 되는지에 대한 예시야
                어떤 문서가 주어지면 어떤 식으로 답변하면 되고 어떤 방식으로 사고를 전개하면 되는지 설명해줄거야
                잘 참고해서 실제 대화에서 퀄리티 높은 답변을 하길바라
                """,
            ),
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                
                너는 보행자와 차량 사고를 전문으로 처리해주는 손해사정사이고 너의 이름은 챗문철이야.
                
                가능하면 bullet point 양식으로 가독성있게 답변해줘
                
                Context: 
                """,
            ),
            (
                "human",
                "보행자와 사고가 났어",
            ),
            (
                "assistant",
                """
                제시 해주신 정보가 부족합니다.
                1. 횡단보도 근처에서 사고나셨나요?
                2. 운전자분께서 확인한 신호상황을 알려주세요.
                3. 보행자분께서 확인한 신호상황을 알려주세요.
                이외에 더 많은 정보를 제시해주시면 더 정확한 안내가 가능합니다.
                """,
            ),
            (
                "human",
                "횡단보도 근처였어 신호등은 따로 없는 길이었고 저녁이고 보행자가 어두운옷을 입고있어서 너무 늦게 봤던거같아",
            ),
            (
                "system",
                """
                어느 정도 정보가 제공된 경우 **반드시** 예상되는 과실비율을 가장 먼저 제시해줘
                아래의 답변은 당연히 예시야 구체적인 값은 실제 상황에서 주어지는 문서를 기반으로 계산해서 알려줘야해
                
                제시된 맥락에 맞는 기본과실 비율도 제시 해줘
                유저가 예상과실 비율과 기본과실비율, 참고사항을 보면 어떻게 이런 계산이 나왔는지 이해할 수 있도록 
                기본과실비율 내용에서 참고사항에 나오는 과실비율을 플러스 또는 마이너스 하면 예상과실비율이 계산되도록!
                
                마지막에 참고사항으로 너가 사고비율을 판단한 근거를 함께 제시해줘
                
                당연히 상황별 변수가 있으니 전문가와 상담하라는 안내를 해줘야해
                
                아래 답변에서 ... 은 너가 채워야하는 부분이야
                
                """,
            ),
            (
                "assistant",
                """
                **예상 과실비율**
                보행자 : ...
                차량 : ...
                
                - 야간 주행하는 경우 ...
                - 보행자의 의무 ...
                
                **기본 과실비율**
                - ...
                
                **참고 사항**
                - ...
                """,
            ),
            (
                "system",
                "EXAMPLE CONVERSATION ENDS",
            ),
            (
                "system",
                "**REAL CONVERSATION STARTS**",
            ),
            (
                "system",
                """
                Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
                
                너는 보행자와 차량 사고를 전문으로 처리해주는 손해사정사이고 너의 이름은 챗문철이야.
                
                가능하면 bullet point 양식으로 가독성있게 답변해줘
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

    message = st.chat_input("질문을 입력해주세요.")

    if message:
        send_message(message, "human")

        with st.chat_message("ai"):
            response = chain_with_history.invoke(
                {"question": message},
                config={"configurable": {"session_id": user_id_manager.uid}},
            )
            # response = chain.invoke(message)
    else:
        st.session_state["messages"] = []


def main_ui():

    st.set_page_config(
        page_title="챗문철",
        page_icon="🎉",
    )

    def get_user_id():
        user_id_manager.increase()
        return True

    # init session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if (
        "user_id_init" not in st.session_state
        or st.session_state["user_id_init"] != True
    ):
        st.session_state["user_id_init"] = True
        user_id_manager.increase()

    st.title("챗문철")

    st.markdown(
        """
    """
    )

    # st.write("user_id", user_id_manager.uid)

    send_message("안녕하세요!", "ai", save=False)

    paint_history()


if __name__ == "__main__":
    load_dotenv()
    main_ui()
    main()
