# Langchain_api_response

Langchain を用いた、質問に対し API キーまたは llm 生成テキストを返すプログラムです。

- document_loading.py
  - まず初めに実行する py ファイル
  - docs/Chroma フォルダ作成し Embedding 結果を格納
  - page_content_column に detail(内容)、metadata に API を格納
- llm_engine.py
  - llm 生成用のコードを用意
  - プロンプト指定
- app.py
  - simirality search で質問文と類似文章の上位検索
  - API キーが格納された metadeta を返答
  - もし該当 API がない（"5. その他" を返答した）場合、get_llm_answer()で llm にテキストをパスする
