
# LlamaIndex RAG-based QA System

`llamaIndex` を使用したQAシステムのサンプル実装です。ユーザーがアップロードしたドキュメントに基づき、チャット形式で応答を生成します。

## ディレクトリ構造

```
01_llama-index
├── data                # 読み込むドキュメントを配置
└── app
    ├── chat_cli.py     # CLI形式のUI
    ├── common_setup.py # LLMおよび設定の読み込み
    └── sample.py       # インデックスの生成およびクエリ処理
```

## セットアップ

1. `data` フォルダにドキュメントを配置してください。
2. `.env` ファイルを `app/secret/` フォルダに作成し、APIキーとモデル情報を設定してください（例: `OPENAI_API_KEY`, `OPENAI_CHAT_MODEL` など）。
3. 必要なパッケージをインストールしてください。

```bash
pip install -r requirements.txt
```

## プログラムの詳細

### 1. `sample.py`
このスクリプトは、`data` フォルダ内のドキュメントを読み込み、インデックスを生成します。また、CLIベースでユーザーのクエリを受け付け、LlamaIndexを使用して回答を生成します。

- **使用方法**
  ```bash
  python sample.py
  ```
  - インデックスを生成し、対話形式でクエリを受け付けます。

- **主な機能**
  - ドキュメントを読み込み、メタデータ（ファイル名、更新日時）を取得。
  - ユーザーのクエリに対してインデックスを用いた検索を行い、回答を表示します。

### 2. `common_setup.py`
環境変数の読み込みとLLM（Large Language Model）の設定を行うモジュールです。`OpenAI`または`Azure OpenAI`のどちらかを選択して利用できます。

- **機能**
  - `.env` ファイルからAPIキーやモデル名を読み込み、`OpenAI` または `Azure OpenAI` のインスタンスを初期化します。
  - モデル設定は、Azureを使用するかどうかのフラグに基づき自動で切り替わります。

### 3. `chat_cli.py`
StreamlitベースのWeb UIです。ユーザーがチャット形式でドキュメントに関する質問を入力し、それに対する回答を表示します。

- **機能**
  - インデックスをセットアップし、ユーザーインターフェースで質問に答えるチャット機能を提供します。
  - ユーザーがインターフェース上で会話の設定をカスタマイズできます（例: 応答のランダム性や使用する言語の選択）。

- **使用方法**
  ```bash
  streamlit run chat_cli.py
  ```

## インデックスの作成方法

まず、`data` フォルダにドキュメントを配置し、以下のコマンドでインデックスを作成します。

```bash
python sample.py
```

## 注意点
- `.env` ファイルには、APIキーやモデルの情報を正確に記述する必要があります。
- インデックスが存在しない場合は、`sample.py` を実行してインデックスを生成してください。

### .envファイルの例

```
# Azureを使用するかどうかのフラグ (trueの場合はAzureを使用、falseの場合はOpenAI APIを直接使用)
USE_AZURE=true

# Azure OpenAI APIキー
AZURE_OPENAI_API_KEY=1234…
# Azure OpenAIのエンドポイント
AZURE_OPENAI_ENDPOINT=https://xxxxx.openai.azure.com/

# Azure OpenAIのエンベディングモデル名
AZURE_EMBEDDING_MODEL=text-embedding-3-small

# Azure OpenAIのチャットモデル名
AZURE_CHAT_MODEL=gpt-4o-mini

# OpenAI APIのバージョン
OPENAI_API_VERSION=2024-08-01-preview

# OpenAI API Key（Azureを使用しない場合に必須）
OPENAI_API_KEY=sk-proj-XXXXXX…
```

## ライセンス
本リポジトリはMITライセンスのもとで公開されています。
