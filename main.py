from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

import os
import chromadb 
import uuid
import shutil # 文件操作模块，为了避免既往数据的干扰，在每次启动时清空 ChromaDB 存储目录中的文件


from dotenv import load_dotenv
load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

DOCUMENT_LOADER_MAPPING = {
    ".pdf": (PDFPlumberLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".xlsx": (UnstructuredExcelLoader, {}),
    ".csv": (CSVLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".xml": (UnstructuredXMLLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
}

def load_document(file_path):
    ext = os.path.splitext(file_path)[1]
    loader_class, loader_args = DOCUMENT_LOADER_MAPPING.get(ext, (None, None))

    if loader_class:
        loader = loader_class(file_path, **loader_args)
        documents = loader.load()
        content = "\n".join([doc.page_content for doc in documents])
        print(f"文档 {file_path} 的部分内容为: {content[:100]}...")
        return content

    print(f"不支持的文档类型: '{ext}'")
    return ""
    
# file_path = 'data/test.pdf'
# load_document(file_path)

def load_embedding_model(model_path='./bge-small-zh-v1.5'):
    print("加载Embedding模型中")
    embedding_model = SentenceTransformer(os.path.abspath(model_path))
    print(f"bge-small-zh-v1.5模型最大输入长度: {embedding_model.max_seq_length}")
    return embedding_model

def embedding_process(folder_path, embedding_model, collection):
    all_chunks = []
    all_ids = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path):
            document_text = load_document(file_path)
            if document_text:
                print(f"文档 {filename} 的总字符数: {len(document_text)}")

                chunks = text_splitter.split_text(document_text)
                print(f"文档 {filename} 分割的文本Chunk数量: {len(chunks)}")

                all_chunks.extend(chunks)
                # 生成每个文本块对应的唯一ID
                all_ids.extend([str(uuid.uuid4()) for _ in range(len(chunks))])

    embeddings = embedding_model.encode(all_chunks, normalize_embeddings=True).tolist()
    # 将文本块的ID、嵌入向量和原始文本块内容添加到ChromaDB的collection中
    collection.add(ids=all_ids, embeddings=embeddings, documents=all_chunks)
    print("嵌入生成完成，向量数据库存储完成.")
    print("索引过程完成.")
    print("********************************************************")

def retrieval_process(query, collection, embedding_model, top_k=6):
    query_embedding = embedding_model.encode(query, normalize_embeddings=True).tolist()

    # 使用向量数据库检索与query最相似的top_k个文本块
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    print(f"查询语句: {query}")
    print(f"最相似的前{top_k}个文本块:")

    retrieved_chunks = []
    # 打印检索到的文本块ID、相似度和文本块信息
    for doc_id, doc, score in zip(results['ids'][0], results['documents'][0], results['distances'][0]):
        print(f"文本块ID: {doc_id}")
        print(f"相似度: {score}")
        print(f"文本块信息:\n{doc}\n")
        retrieved_chunks.append(doc)

    print("检索过程完成.")
    print("********************************************************")
    return retrieved_chunks

def generate_process(query, chunks):

    context = ""
    for i, chunk in enumerate(chunks):
        context += f"参考文档{i+1}: \n{chunk}\n\n"

    prompt = f"根据参考文档回答问题：{query}\n\n{context}"
    print(f"生成模型的Prompt: {prompt}")

    messages = [{'role': 'user', 'content': prompt}]


    model = ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-max",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    )

    response = model.invoke(messages)
    print(response)


def main(clear_db: bool = False):
    print("RAG过程开始.")

    # 为了避免既往数据的干扰，在每次启动时清空 ChromaDB 存储目录中的文件
    chroma_db_path = os.path.abspath("./chroma_db")
    if clear_db and os.path.exists(chroma_db_path):
        shutil.rmtree(chroma_db_path)

    # 创建ChromaDB本地存储实例和collection
    client = chromadb.PersistentClient(chroma_db_path)
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"} 
    ) 

    embedding_model = load_embedding_model()
    # 检查是否需要重新索引
    if collection.count() == 0:
        print("开始索引文档")
        embedding_process('./data', embedding_model, collection)
    else: 
        print(f"使用现有索引，已有 {collection.count()} 个文档块")

    query = "下面报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
    retrieval_chunks = retrieval_process(query, collection, embedding_model)
    # print(retrieval_chunks[0])

    if retrieval_chunks:
        generate_process(query, retrieval_chunks)
    
    print("RAG过程结束.")

if __name__ == "__main__":
    main()