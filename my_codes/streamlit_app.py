import os
import sys

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv, find_dotenv

from ark_embedding import ArkEmbeddings


_ = load_dotenv(find_dotenv('.env.local'))
# 设置大模型参数
api_key = os.getenv("DEEPSEEK_API_KEY")
api_url = os.getenv("DEEPSEEK_API_URL")
model = os.getenv("DEEPSEEK_MODEL")
# 设置向量数据库参数
embedding_api_key = os.getenv("ARK_API_KEY")
embedding_api_url = os.getenv("ARK_API_URL")
embedding_model = os.getenv("ARK_EMBEDDING_MODEL")

def get_retriever():
    """
    获取向量数据库的检索器

    Returns:
        vectordb: 向量数据库的检索器
    """
    # 初始化 Embeddings
    embedding = ArkEmbeddings(
        api_key=embedding_api_key,
        api_url=embedding_api_url,
        model=embedding_model,
    )

    # 向量数据库持久化路径
    persist_directory = "../data_base/vector_db/chroma"

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding,
    )

    return vectordb

def combine_docs(docs):
    """
    整合知识库的文档

    Args:
        docs: 知识库的文档

    Returns:
        str: 整合后的知识库文档
    """
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    """
    获取问答链

    Returns:
        qa_history_chain: 问答链
    """
    retriever = get_retriever()
    llm = ChatOpenAI(
        openai_api_key=api_key,
        base_url=api_url,
        model_name=model,
        temperature=0.0,
    )

    # 压缩问题的系统 prompt
    condense_question_system_template = (
        "请根据聊天记录完善用户最新的问题，"
        "如果用户最新的问题不需要完善则返回用户的问题。"
        "确保最后的输出是一个完整的问题。"  # NOTE: 这里需要确保最后的输出是一个完整的问题
    )

    # 构建压缩问题的 prompt template
    condense_question_prompt = ChatPromptTemplate([
        ("system", condense_question_system_template),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])

    # 构造“总结历史信息”的检索文档的处理链
    # RunnableBranch 会根据条件选择要运行的分支
    retrieve_docs = RunnableBranch(
        # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
        (lambda x: not x.get("chat_history", ""), (lambda x: x["input"]) | retriever, ),
        # 分支 2 : 若聊天记录中有 chat_history 则先让 llm 根据聊天记录完善问题再查询向量数据库
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    # 问答链的系统prompt
    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )

    # 制定prompt template
    qa_prompt = ChatPromptTemplate(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    # 定义“整合知识库”的问答链
    # 1. 整合知识库信息进入context
    # 2. 拼装prompt, 整合context和chat_history进入qa_chain
    # 3. 请求llm
    # 4. 格式化输出
    qa_chain = (
        RunnablePassthrough.assign(context=combine_docs) # 使用 combine_docs 函数，整合知识库的内容得到context输入 qa_prompt
        | qa_prompt # 问答模板
        | llm
        | StrOutputParser() # 规定输出的格式为 str
    )

    # 定义带有历史记录的问答链
    # 1. 检索知识库(结合总结历史信息), 并存入context
    # 2. 整合context和chat_history进入qa_chain
    qa_history_chain = RunnablePassthrough.assign(
        context=retrieve_docs # 将查询结果存为 context
    ).assign(answer=qa_chain) # 将最终结果存为 answer

    return qa_history_chain

def gen_response(chain, input, chat_history):
    """
    生成响应

    Args:
        chain: 问答链
        input: 用户输入
        chat_history: 聊天记录

    Returns:
        response: 响应
    """
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

def main():
    st.markdown("### 🦜🔗 动手学大模型应用开发")
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立窗口 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages:
        with messages.chat_message(message[0]):
            st.write(message[1])
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages,
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))
