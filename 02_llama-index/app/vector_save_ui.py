#vector_save_ui.py
import streamlit as st
import subprocess
import sys
import os
import time
import re
import json
from datetime import datetime
import pytz

# vector_save.pyのパスを設定
VECTOR_SAVE_PATH = "app/vector_save.py"  # 適切なパスに変更してください
METADATA_FILE = "storage/metadata.json"  # メタデータファイルのパス

def run_vector_save(args):
    command = [sys.executable, VECTOR_SAVE_PATH] + args
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    return process

def safe_get_keywords(info):
    keywords = info.get('keywords', [])
    if isinstance(keywords, list):
        return keywords
    elif isinstance(keywords, dict):
        return list(keywords.values())[0] if keywords else []
    elif isinstance(keywords, str):
        return [kw.strip() for kw in keywords.split(',')]
    else:
        return []
    
def process_output(process, progress_bar, status_text):
    total_files = 0
    processed_files = 0
    output = []
    for line in iter(process.stdout.readline, ''):
        output.append(line)
        if "Updated/Inserted document:" in line:
            processed_files += 1
            if total_files > 0:
                progress = min(processed_files / total_files, 1.0)
                progress_bar.progress(progress)
            status_text.text(f"処理中: {line.strip()}")
        elif "新しいインデックスを作成します" in line:
            status_text.text("新しいインデックスを作成中...")
        elif match := re.search(r"with (\d+) chunks", line):
            total_files += int(match.group(1))
    
    progress_bar.progress(1.0)
    status_text.text("処理完了")
    time.sleep(1)  # 「処理完了」メッセージを表示するための短い遅延
    status_text.empty()  # ステータステキストをクリア
    
    error_output = process.stderr.read()
    
    return ''.join(output), error_output

def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def format_datetime(timestamp):
    return datetime.fromtimestamp(timestamp, pytz.timezone('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')

st.title("Vector Save Configuration")

st.write("このアプリケーションは、vector_save.pyの設定を行い、実行します。")

# メタデータの読み込み
metadata = load_metadata()

# 各オプションの設定
chunk_size = st.number_input("Chunk Size", min_value=1, value=4096, help="テキスト分割時のチャンクサイズ")
chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=chunk_size-1, value=min(40, chunk_size-1), help="チャンク間のオーバーラップ")

extract_keywords = st.checkbox("キーワード抽出", help="ドキュメントからキーワードを抽出します。自動検索でキーワード検索を使用する際に有効。")
num_keywords = st.number_input("抽出するキーワード数", min_value=1, value=10, disabled=not extract_keywords)

summarize = st.checkbox("サマリー生成", help="ドキュメントのサマリーを生成します。再帰的検索時に有効。")
summary_level = st.radio("サマリーレベル", ["file", "chunk"], disabled=not summarize)

force_update = st.checkbox("強制更新", help="既存のインデックスを無視して強制的に再作成します")

# 実行ボタン
if st.button("Vector Save を実行"):
    if chunk_overlap >= chunk_size:
        st.error("チャンクオーバーラップはチャンクサイズより小さくなければなりません。")
    else:
        args = [
            f"--chunk-size={chunk_size}",
            f"--chunk-overlap={chunk_overlap}"
        ]
        if extract_keywords:
            args.append("--extract-keywords")
            args.append(f"--num-keywords={num_keywords}")
        
        if summarize:
            args.append("--summarize")
            args.append(f"--summary-level={summary_level}")
        
        if force_update:
            args.append("--force-update")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        process = run_vector_save(args)
        output, error_output = process_output(process, progress_bar, status_text)
        
        if "ERROR" in error_output or "Exception" in error_output or "Traceback" in error_output:
            st.error("エラーが発生しました:")
            st.code(error_output)
        else:
            st.success("実行が正常に完了しました")
                
        # 常に出力を表示
        st.subheader("実行ログ:")
        st.code(output)

        # メタデータを再読み込み
        metadata = load_metadata()

# メタデータの表示
if metadata:
    st.subheader("metadata.json")
    
    # テーブルデータの準備部分
    table_data = []
    for filename, info in metadata.items():
        keywords = safe_get_keywords(info)
        row = {
            "ファイル名": filename,
            "最終更新日時": format_datetime(info.get('modification_time', 0)),
            "チャンク数": info.get('chunks', 0),
            "合計文字数": info.get('total_chars', 0),
            "ファイルサマリー": info.get('summary', '')[:50] + '...' if info.get('summary') else '',
            "キーワード": ', '.join(keywords[:5]) + ('...' if len(keywords) > 5 else '') if keywords else '',
            "チャンクサマリー（最初の3つ）": '; '.join(info.get('chunk_summaries', [])[:3]) + ('...' if len(info.get('chunk_summaries', [])) > 3 else '')
        }
        table_data.append(row)
    
    # テーブルの表示
    st.table(table_data)
else:
    st.warning("metadata.jsonが見つかりません。Vector Saveを実行してデータを生成してください。")
