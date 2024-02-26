from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# LLMモデルの設定
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_llm_answer(query):
    prompt_template = """# 命令書:
    あなたはスマートホームの優秀なオペレーターで、家電を操作するAPIキーの発行が可能です。
    以下の制約条件をもとに、ユーザーからの問い合わせに対して最高の回答文を出力してください。

    # 制約条件:
    ・与えられた参考情報の中から重要なキーワードを取り残さないでください。
    ・参考情報から答えがわからない場合、知りませんと答えて、それ以上回答を生成しようとしないでください。
    ・回答文は簡潔にしてください。

    # 問い合わせ：
    {question}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    # LangChain Expression Language (LCEL) Chainの記法
    chain = prompt | llm
    answer = chain.invoke({"question": query})
    return answer.content

# retrievalQAを用いる場合
# llmに自由生成させるのではなく、特定のdb内容から生成させたい際は以下関数を使用。
def get_retrievalQA_answer(query, db):
    # PromptTemplateの作成
    prompt_template_qa = """# 命令書:
    あなたはスマートホームの優秀なオペレーターで、家電を操作するAPIキーの発行が可能です。
    以下の制約条件をもとに、ユーザーからの問い合わせに対して最高の回答文を出力してください。

    # 制約条件:
    ・与えられた参考情報の中から重要なキーワードを取り残さないでください。
    ・参考情報から答えがわからない場合、知りませんと答えて、それ以上回答を生成しようとしないでください。
    ・回答文は簡潔にしてください。

    # 参考情報：
    {context}

    # 問い合わせ：
    {question}

    # 回答文："""
    prompt_qa = PromptTemplate(template=prompt_template_qa, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt_qa}
    retriever = db.as_retriever(search_kwargs={"k": 1})
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
    )
    answer = qa.invoke(query)
    return answer["result"]
