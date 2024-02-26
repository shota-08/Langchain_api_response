# streamlit run app.py をターミナルで実行

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from llm_engine import get_llm_answer
# from llm_engine import get_retrievalQA_answer

st.set_page_config(page_title='Langchain x llm')
st.title('Langchain x llm')

# 初期化
embedding_function = OpenAIEmbeddings()

# ローカルdb呼び出し
persist_directory = "./docs/chroma"
db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

# 質問文
query = st.chat_input('質問文を入力')

# チャットログセッションの初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

if query:
    docs = db.similarity_search(query, k=3)
    answer_ref = docs[0].metadata
    answer = answer_ref['API']

    if answer == "5. その他":
        answer = get_llm_answer(query)
        # answer = get_retrievalQA_answer(query, db)

    # ログの表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # chat表示
    with st.chat_message(USER_NAME):
        st.write(query)

    with st.chat_message(ASSISTANT_NAME):
        st.write(answer)

    # data表示
    with st.expander('docs', expanded=False):
        st.write(docs)

    # セッションにログを追加
    st.session_state.chat_log.append({"name" : USER_NAME, "msg" : query})
    st.session_state.chat_log.append({"name" : ASSISTANT_NAME, "msg" : answer})