from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os

HF_TOKEN = os.environ.get("HF_TOKEN")
print("Token loaded:", HF_TOKEN is not None, "| length:", len(HF_TOKEN) if HF_TOKEN else 0)

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    max_new_tokens=100,
)
chat = ChatHuggingFace(llm=llm)
print(chat.invoke("What is AI?"))