# streamlit run document_loading.py をターミナルで実行
# streamlit が環境変数を繋いでくれるため、.envファイル不要

import streamlit as st
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# csvから読み込むコーディングも可能
# 空のListを用意し、APIカラムとdetailカラムをそれぞれfor文で格納する
data = [
    {"API" : "1. 照明用API" , "detail" : "部屋の電気を付ける、または消すためのAPIキーを発行します"},
    {"API" : "2. カーテン用API" , "detail" : "部屋のカーテンを開ける、または閉めるためのAPIキーを発行します"},
    {"API" : "3. エアコンAPI" , "detail" : "部屋のエアコンを付ける、または消すためのAPIキーを発行します"},
    {"API" : "4. お風呂API" , "detail" : "お風呂の給湯をするためのAPIキーを発行します"},
    {"API" : "5. その他" , "detail" : "あいさつや問い合わせに対応します"},
]

df = pd.DataFrame(data)

# page_content_columnでcontentになるカラムを指定、metadataに'API'を格納
loader = DataFrameLoader(df, page_content_column='detail')

documents = loader.load()

# 初期化
embedding_function = OpenAIEmbeddings()

# 以下のフォルダにEmbeddingデータを保存
persist_directory = "docs/chroma"
db = Chroma.from_documents(documents, embedding_function, persist_directory=persist_directory)