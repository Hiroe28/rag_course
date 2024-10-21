
# LlamaIndex QA System Project

このプロジェクトは、LlamaIndexを使用したQAシステムの2つのサンプル実装を提供します。`01_llama-index` 、`02_llama-index` はWebベースのUIを備えています。どちらも、ユーザーがアップロードしたドキュメントに基づいて質問に対する回答を生成するRetrieval-Augmented Generation (RAG) システムです。

## ディレクトリ構造

```
LlamaIndex-QA-System
├── 01_llama-index      # 基本的なLlamaIndexプログラム
└── 02_llama-index      # 応用的なLlamaIndexプログラム
```

## セットアップ

プロジェクト全体のセットアップ手順は以下の通りです。

1. リポジトリをクローン、またはコードをダウンロードします。

2. 必要なパッケージをインストールします。

   ```bash
   pip install -r requirements.txt
   ```

3. `.env` ファイルを `app/secret/` フォルダに作成し、APIキーとモデル情報を設定してください（例: `OPENAI_API_KEY`, `AZURE_OPENAI_API_KEY` など）。

## 各ディレクトリの説明

### 01_llama-index

`01_llama-index` は、ユーザーがドキュメントに対する質問を入力し、回答を生成する基本的なシステムです。

#### ディレクトリ構造

```
01_llama-index
├── data                # 読み込むドキュメントを配置
└── app
    ├── chat_ui.py     # UI（単体で実行可能）
    ├── common_setup.py # LLMおよび設定の読み込み
    └── sample.py       # インデックスの生成およびクエリ処理
```

#### 使い方

1. `data` フォルダにドキュメントを配置します。
2. `.env` ファイルを作成し、APIキーやモデル情報を設定します。
3. 以下のコマンドでCLIを起動します。

   ```bash
   python sample.py
   ```

   - インデックスが生成され、ユーザーのクエリに対する回答がCLI上に表示されます。

### 02_llama-index

`02_llama-index` は、Streamlitを使用したWeb UIを備えており、ユーザーがブラウザ上でチャット形式で質問を行い、回答を得ることができるシステムです。

#### ディレクトリ構造

```
02_llama-index
├── data                # 読み込むドキュメントを配置
└── app
    ├── chat_ui.py     # RAGのUIツール
    ├── common_setup.py # LLMおよび設定の読み込み
    ├── rag.py          # RAGのセットアップおよびカスタムリトリーバー
    ├── vector_save.py  # インデックスの生成および保存
    └── vector_saveUI.py  # vector_save.pyのUIツール
```

#### 使い方

1. `data` フォルダにドキュメントを配置します。
2. `.env` ファイルを作成し、APIキーやモデル情報を設定します。
3. 02_llama-indexで実行vector_save.pyかvector_saveUI.pyを実行し、ベクトルインデックスを生成します。
   ```bash
   python app\vector_save.py
   ```
   ```bash
   streamlit run app\vector_saveUI.py
   ```
4. 02_llama-indexで、以下のコマンドでWeb UIを起動します。

   ```bash
   streamlit run app\chat_ui.py
   ```

   - ブラウザが開き、ユーザーはWeb UIを介してドキュメントに基づく質問をできます。

## 環境変数設定

両方のシステムでは、APIキーやモデル設定のために`.env`ファイルが必要です。以下は例です。

```
# Azureを使用するかどうかのフラグ (trueの場合はAzureを使用、falseの場合はOpenAI APIを直接使用)
USE_AZURE=true

# Azure OpenAI APIキー
AZURE_OPENAI_API_KEY=1234
# Azure OpenAIのエンドポイント
AZURE_OPENAI_ENDPOINT=https://xxxxx.openai.azure.com/

# Azure OpenAIのエンベディングモデル名
AZURE_EMBEDDING_MODEL=text-embedding-3-small

# Azure OpenAIのチャットモデル名
AZURE_CHAT_MODEL=gpt-4o-mini

# OpenAI APIのバージョン
OPENAI_API_VERSION=2024-08-01-preview

# OpenAI API Key（Azureを使用しない場合に必須）
OPENAI_API_KEY=sk-proj-XXXXXX
```



