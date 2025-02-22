import os
import csv
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from src.prompt import *

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

def file_preprocessing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = "".join(page.page_content for page in data)

    document_ques_gen = [Document(page_content=question_gen)]
    splitter_ans_gen = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)
    
    return document_ques_gen, document_answer_gen

def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = file_preprocessing(file_path)

    llm_ques_gen_pipeline = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])
    REFINE_PROMPT_QUESTIONS = PromptTemplate(template=refine_template, input_variables=["existing_answer", "text"])

    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    ques = ques_gen_chain.run(document_ques_gen)
    ques_list = [q.strip() for q in ques.split("\n") if q.endswith("?") or q.endswith(".")]

    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)
    
    llm_ans_gen = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_ans_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return answer_generation_chain, ques_list

def get_csv(file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    
    base_folder = "static/output/"
    os.makedirs(base_folder, exist_ok=True)

    output_file = os.path.join(base_folder, "QA.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])

        for question in ques_list:
            answer = answer_generation_chain.run(question)
            csv_writer.writerow([question, answer])
    
    return output_file
