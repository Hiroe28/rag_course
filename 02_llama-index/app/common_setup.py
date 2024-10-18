import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

def load_environment():
    env_path = Path(__file__).parent / "secret" / ".env"
    load_dotenv(env_path)    
    
    config = {
        "use_azure": os.getenv("USE_AZURE", "false").lower() == "true",
        "api_key": os.getenv("AZURE_OPENAI_API_KEY") if os.getenv("USE_AZURE", "false").lower() == "true" else os.getenv("OPENAI_API_KEY"),
        "chat_model": os.getenv("AZURE_CHAT_MODEL") if os.getenv("USE_AZURE", "false").lower() == "true" else os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("AZURE_EMBEDDING_MODEL") if os.getenv("USE_AZURE", "false").lower() == "true" else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    }
    
    if config["use_azure"]:
        config.update({
            "endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "api_version": os.getenv("OPENAI_API_VERSION", "2024-08-01-preview"),
        })
    
    return config

def get_llm_and_embed_model():
    config = load_environment()
    
    if config["use_azure"]:
        llm = AzureOpenAI(
            model=config["chat_model"],
            deployment_name=config["chat_model"],  # モデル名をデプロイメント名として使用
            api_key=config["api_key"],
            azure_endpoint=config["endpoint"],
            api_version=config["api_version"]
        )
        embed_model = AzureOpenAIEmbedding(
            model=config["embedding_model"],
            deployment_name=config["embedding_model"],  # モデル名をデプロイメント名として使用
            api_key=config["api_key"],
            azure_endpoint=config["endpoint"],
            api_version=config["api_version"]
        )
    else:
        llm = OpenAI(model=config["chat_model"], api_key=config["api_key"])
        embed_model = OpenAIEmbedding(model=config["embedding_model"], api_key=config["api_key"])
    
    return llm, embed_model
