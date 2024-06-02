from langchain_community.document_loaders import AmazonTextractPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCT
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
import boto3
import os
from dotenv import load_dotenv


load_dotenv()

mode = os.getenv("MODE")


if mode == "local":
    documents = PyPDFDirectoryLoader(os.getenv("LOCAL_FILE_PATH")).load()
else:
    textract_client = boto3.client(
        "textract", region_name=os.getenv("AWS_REGION_NAME"), 
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")    
    )

    file_path = os.getenv("AWS_FILE_PATH")
    loader = AmazonTextractPDFLoader(file_path, client=textract_client)
    documents = loader.load()

splitter = RCT(
    chunk_size=20000,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False
)

pages = splitter.split_documents(documents)

def gen_page_ids(pages):
    page_index = None
    current_page_id = 0

    for p in pages:
        source = p.metadata["source"]
        page = p.metadata["page"]
        current_page_index = f"{source}:{page}"

        if current_page_index == page_index:
            current_page_id += 1
        else:
            current_page_id = 0

        page_index = current_page_index

        page_id = f"{current_page_index}:{current_page_id}"

        p.metadata["id"] = page_id

    return pages

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embedding_function = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


def add_to_chroma(pages):

    db = Chroma(persist_directory="manualstore", embedding_function = embedding_function)

    items_in_db = db.get(include=[])

    items_in_db_ids = set(items_in_db["ids"])

    if len(items_in_db_ids):
        print(f"We have {len(items_in_db_ids)} documents in the database")

    new_documents = []
    
    pages_with_ids = gen_page_ids(pages)
    for page in pages_with_ids:
        page_id = page.metadata["id"]
        if page_id not in items_in_db_ids:
            new_documents.append(page)


    if len(new_documents):
        print(f"We are adding {len(new_documents)} documents to the database")

        new_documents_ids = [page.metadata["id"] for page in new_documents]

        db.add_documents(new_documents, ids=new_documents_ids)

    else:
        print("No new documents to add")

    return db

vectorstore = add_to_chroma(pages)

retriever = vectorstore.as_retriever()

model = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""


contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

qa_system_prompt = """
    You are an intelligent assistant. Answer the questions based solely on the provided context. Do not use any outside knowledge or offer information not included in the context. If the answer is not available within the provided context, respond with "The information is not available in the provided context"
{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)


qa_chain = create_stuff_documents_chain(model, qa_prompt)

retrieval_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

store = {}

def get_session_ids(session_id:str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


chain = RunnableWithMessageHistory(
    retrieval_chain,
    get_session_ids,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)


