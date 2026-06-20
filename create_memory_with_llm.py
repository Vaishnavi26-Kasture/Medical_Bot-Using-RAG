from dotenv import load_dotenv
load_dotenv()

import os

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID = "Qwen/Qwen2.5-7B-Instruct"


def load_llm(huggingface_repo_id):

    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        temperature=0.5,
        max_new_tokens=512
    )

    chat_model = ChatHuggingFace(llm=llm)
    return chat_model


CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.

If you don't know the answer, just say that you don't know.

Do not make up answers.

Context:
{context}

Question:
{question}

Start the answer directly.
"""


def set_custom_prompt(custom_prompt_template):

    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

    return prompt


DB_FAISS_PATH = "vectorstore/db_faiss"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    DB_FAISS_PATH,
    embedding_model,
    allow_dangerous_deserialization=True
)

qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(HUGGINGFACE_REPO_ID),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)

user_query = input("Write Query Here: ")

response = qa_chain.invoke(
    {"query": user_query}
)

print("\nRESULT:\n")
print(response["result"])

print("\nSOURCE DOCUMENTS:\n")
print(response["source_documents"])