
# LlamaIndex Project

このプロジェクトは、LlamaIndexを使用してドキュメントのインデックスを作成し、ユーザーの質問に対する回答を生成するシンプルなチャットシステムです。以下のディレクトリ構造を持っています。

```
01_llama-index：カレントディレクトリ
└ data … 読み込むドキュメントを配置
└ app
    └ chat_cli.py … UI
    └ common_setup.py … secret\.envを読み込みLLMを設定
    └ sample.py … dataフォルダのドキュメントに対してインデックス作成
```

## プログラム概要

### 1. chat_cli.py
`chat_cli.py` はStreamlitを使用して、シンプルなチャットインターフェースを提供します。以下が主な機能です。

- **UI設定**: Streamlitを用いて、ユーザーインターフェースを作成します。ページのタイトルやレイアウト、サイドバーの設定が含まれます。
- **インデックスの設定**: `setup_index`メソッドで、`data`ディレクトリ内のドキュメントを読み込み、LlamaIndexを使用してインデックスを作成します。
- **会話機能**: ユーザーからの入力に基づいて、LlamaIndexを使用して質問に回答します。回答はリアルタイムで生成され、チャット形式で表示されます。

詳細なコードは以下の通りです。
```python
# chat_cli.pyの一部
def setup_index(self):
    llm, embed_model = get_llm_and_embed_model()
    documents = SimpleDirectoryReader(self.data_dir, recursive=True).load_data()
    self.index = VectorStoreIndex.from_documents(documents)
```

### 2. common_setup.py
`common_setup.py` は、環境変数からLLM（Large Language Model）とエンベディングモデルを設定するためのファイルです。

- **環境設定の読み込み**: `.env`ファイルからAPIキーやモデル名を読み込み、AzureまたはOpenAIのサービスを設定します。
- **モデルの取得**: `get_llm_and_embed_model`関数で、LLMとエンベディングモデルのインスタンスを作成し、返します。

以下は一部のコードです。
```python
def get_llm_and_embed_model():
    config = load_environment()
    if config["use_azure"]:
        llm = AzureOpenAI(model=config["chat_model"], api_key=config["api_key"])
    else:
        llm = OpenAI(model=config["chat_model"], api_key=config["api_key"])
    return llm, embed_model
```

### 3. sample.py
`sample.py` は、`data`フォルダ内のドキュメントに対してインデックスを作成するスクリプトです。

- **インデックスのセットアップ**: `setup_index`関数で、データディレクトリからドキュメントを読み込み、インデックスを作成します。
- **ユーザーのクエリ処理**: インデックスに対してユーザーからのクエリを受け付け、回答を生成します。

以下はサンプルコードです。
```python
def setup_index():
    reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True)
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index
```

## 使用方法

1. `.env`ファイルを`secret`ディレクトリ内に作成し、APIキーやモデル情報を設定します。
2. `chat_cli.py`を実行して、チャットUIを起動します。
3. `sample.py`を使って、`data`ディレクトリ内のドキュメントに対するインデックスを作成し、CLIからクエリを入力して回答を確認できます。

## 必要な環境変数

`.env`ファイルには以下の情報を設定する必要があります。
- `USE_AZURE`: Azureサービスを使用する場合は`true`、OpenAIを使用する場合は`false`。
- `OPENAI_API_KEY`または`AZURE_OPENAI_API_KEY`: APIキーを設定します。
- 使用するモデル名やエンベディングモデルの設定も必要です。

## 注意点

- `data`ディレクトリには、読み込みたいドキュメントを配置してください。
- Python環境には、必要なライブラリ（`llama_index`, `streamlit`, `dotenv`など）をインストールする必要があります。
