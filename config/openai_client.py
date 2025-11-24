# config/openai_client.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

DEFAULT_MODEL = "gpt-5.1"  # 主力推理模型
CHEAP_MODEL = "gpt-5-mini"  # 摘要/记忆压缩用

