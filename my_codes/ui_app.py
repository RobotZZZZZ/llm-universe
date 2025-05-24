## 使用 chainlit 框架实现流式输出
## 运行命令
## chainlit run my_codes/ui_app.py
## 停止命令
## pkill -f "chainlit run"

import os
import sys

import chainlit

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Chroma

from dotenv import load_dotenv, find_dotenv

from ark_embedding import ArkEmbeddings

_ = load_dotenv(find_dotenv('.env.local'))

# 在开头添加环境变量检查
required_vars = ["DEEPSEEK_API_KEY", "DEEPSEEK_API_URL", "DEEPSEEK_MODEL", 
                 "ARK_API_KEY", "ARK_API_URL", "ARK_EMBEDDING_MODEL"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"缺少环境变量: {', '.join(missing_vars)}")

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
        retriever: 向量数据库的检索器
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

    # 将向量数据库转换为检索器
    retriever = vectordb.as_retriever()
    return retriever

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

    # 构造"总结历史信息"的检索文档的处理链
    # RunnableBranch 会根据条件选择要运行的分支
    retrieve_docs = RunnableBranch(
        # 分支 1: 若聊天记录中没有 chat_history 则直接使用用户问题查询向量数据库
        (lambda x: not x.get("chat_history", ""), RunnableLambda(lambda x: x["input"]) | retriever, ),
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

    # 定义"整合知识库"的问答链
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

@chainlit.on_chat_start
async def start():
    """
    在聊天开始时初始化
    """
    # 初始化问答链
    qa_history_chain = get_qa_history_chain()
    
    # 将问答链存储到用户会话中
    chainlit.user_session.set("qa_history_chain", qa_history_chain)
    chainlit.user_session.set("messages", [])
    
    await chainlit.Message(content="你好！我是基于知识库的问答助手，请问有什么问题吗？").send()

@chainlit.on_message
async def main(message: chainlit.Message):
    """
    处理用户消息
    """
    # 获取用户会话中的问答链和历史消息
    qa_history_chain = chainlit.user_session.get("qa_history_chain")
    messages = chainlit.user_session.get("messages", [])
    
    # 获取用户输入
    user_input = message.content
    
    # 准备响应消息
    msg = chainlit.Message(content="")
    await msg.send()
    
    # 生成回复（流式输出）
    response_text = ""
    
    # 获取流式响应
    response_stream = qa_history_chain.stream({
        "input": user_input,
        "chat_history": messages
    })
    
    for chunk in response_stream:
        if "answer" in chunk:
            # 获取增量内容
            chunk_content = chunk["answer"]
            response_text += chunk_content
            
            # 更新消息内容
            msg.content = response_text
            await msg.update()
    
    # 将对话添加到历史记录
    messages.append(("user", user_input))
    messages.append(("assistant", response_text))
    
    # 更新用户会话
    chainlit.user_session.set("messages", messages)
