from dotenv import load_dotenv
import PyPDF4
import pdfplumber
import langchain
import re 
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage


def extract_metadata_from_pdf(file_path):
    """ Extracts metadata from a PDF file and returns it as a dictionary. """

    with open(file_path, 'rb') as pdf_file:
        reader = PyPDF4.PdfFileReader(pdf_file)
        metadata = reader.getDocumentInfo()
        return {
            'title': metadata.get("/Title", "").strip(),
            'author': metadata.get("/Author", "").strip(),
            'creation_date': metadata.get("/CreationDate", "").strip()
        }


def extract_text_from_pdf(file_path):
    """ Extracts text from a PDF file and returns it as a string. """
    
    with pdfplumber.open(file_path) as pdf_file:
        pages = []
        for page_number, page in enumerate(pdf_file.pages):
            text = page.extract_text()
            if text.strip():
                pages.append((page_number + 1, text))

    return pages 


def parse_pdf(file_path):
    """ Extracts metadata, text from pdf file and returns a tuple. """
    
    metadata = extract_metadata_from_pdf(file_path)
    pages = extract_text_from_pdf(file_path)

    return (pages, metadata )


def merge_hyphenated_words(text):
    """ Merges hyphenated words in a string. """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    



def fix_newlines(text):
    """ Fixes newlines in a string. """
    return re.sub(r"(?<!\n)\n(?!\n)", " ", text)


def remove_multiple_newlines(text):
    """ Removes multiple newlines in a string. """
    return re.sub(r"\n{2,}", "\n", text)

def preprocess_pages(pages, cleaning_functions):
    cleaned_pages = []
    for page_number , text in pages:
        for cleaning_function in cleaning_functions:
            text = cleaning_function(text)
        cleaned_pages.append((page_number, text))
    return cleaned_pages

def text_to_docs(text, metadata, chunk_size=1000, chunk_overlap=200):
    doc_chuncks = []

    for page_number, page_text in text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, 
                                                       separators= ["\n\n", "\n", ".", "!", "?", ",", " ", ""], chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(page_text)
        for i , chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata= {
                    "page_number": page_number,
                    "chunk": i,
                    "source": f"p{page_number}-{i}",
                    **metadata
                },
            )
            doc_chuncks.append(doc)
    return doc_chuncks


def make_chain(model, vector_store):
    return ConversationalRetrievalChain.from_llm(
        model,
        retriever=vector_store.as_retriever(search_type='similarity'),
        return_source_documents=True,
        #verbose=True
    )

# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004

chat_history = []

def clear_history():
    chat_history = []



if __name__ == "__main__":

    load_dotenv()
    
    st.subheader("Lanchain Question Answering Application")
    with st.sidebar:
        #get api key using streamlit text input
        OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type="password")
        if OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        #file uploader widget
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"]) 

        #chunk size widget
        chunk_size = st.number_input('Chunk size:', min_value=512, max_value=2048, value=1000, on_change=clear_history)

        #index name
        index_name = st.text_input('Collection name:', type="default")

        #add data button widget
        add_data = st.button("Add data", on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner("Reading, chunking and embedding file...."):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                #parse pdf
                pages, metadata = parse_pdf(file_name)
                #preprocess pages
                cleaning_functions = [merge_hyphenated_words, fix_newlines, remove_multiple_newlines]
                cleaned_pages = preprocess_pages(pages, cleaning_functions)
                #text to docs
                doc_chunks = text_to_docs(cleaned_pages, metadata, chunk_size=chunk_size)

                #document chunk size
                doc_chunks = doc_chunks[:70]

                st.write(f'Document Title: {metadata["title"]}')
                st.write(f'Chunk size: {chunk_size}, chunks: {len(doc_chunks)}')

                total_tokens, cost = calculate_embedding_cost(doc_chunks)
                st.write(f'Total Tokens: {total_tokens}, Embedding Cost in USD: {cost:.6f}')
                

                # generate embedding 
                OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                
                path = os.path.join('data', index_name)
                

                vector_store = Chroma.from_documents(
                    doc_chunks,
                    embeddings,
                    collection_name=index_name,
                    persist_directory="data/" + index_name
                )

                vector_store.persist()

                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully.')
    question = st.text_input('Ask a question about the content of your file:')
    if question:
        with st.spinner("Searching for answer...."):
            if 'vs' in st.session_state:
                vector_store = st.session_state.vs
                model = ChatOpenAI(
                        model_name="gpt-3.5-turbo",
                        temperature="0",
                        # verbose=True
                    )

                
                #make chain
                chain = make_chain(model, vector_store)
            
                response = chain({"question": question, "chat_history": chat_history })

                answer = response["answer"]
                source = response["source_documents"]
                chat_history.append(HumanMessage(content=question))
                chat_history.append(AIMessage(content=answer))
                

                # text area widget for the LLM answer
                st.text_area('LLM Answer:  ', value=answer, height=200)

                st.write("\n\nSources:\n")
                for document in source:
                    st.write(f"Page: {document.metadata['page_number']}")
                    st.write(f"Text chunk: {document.page_content[:160]}...\n")
                

                
    
