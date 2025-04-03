from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from pypdf import PdfReader

def process_text(text):
    # processing text by splitting into chunks
    # then converting them into embeddings to form
    # a knowledge base

    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200, #overlap between chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text) #splitting into chunks

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    knowledgebase = FAISS.from_texts(chunks, embeddings)

    return knowledgebase

def summarizer(pdf):

    pdf_reader = PdfReader(pdf)
    text = ''

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    knowledgebase = process_text(text)

    query = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences."

    if query:
        # perform similarity search in knowledge base using query
        docs = knowledgebase.similarity_search(query)

        OpenAIModel = "gpt-3.5-turbo-16k"
        llm = ChatOpenAI(model=OpenAIModel, temperature=0.8)

        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cost:
            response = chain.run(input_documents=docs, question=query)
            print(cost)
            return response





