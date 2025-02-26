import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
load_dotenv()
        
def prepare_rag_llm(
    token, vector_store_list, temperature, max_length
):
    # Load embeddings instructor
    instructor_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

    # Load db
    loaded_db = FAISS.load_local(
        f"vector store/{vector_store_list}", instructor_embeddings, allow_dangerous_deserialization=True
    )

    # Load LLM
    llm = HuggingFaceHub(
        repo_id='tiiuae/falcon-7b-instruct',
        model_kwargs={"temperature": temperature, "max_length": max_length},
        huggingfacehub_api_token=token
    )

    memory = ConversationBufferWindowMemory(
        k=2,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    # Create the chatbot with a system instruction
    qa_conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=loaded_db.as_retriever(search_kwargs={"k": 9}),
        return_source_documents=True,
        memory=memory,
    )

    return qa_conversation

def generate_answer(question, token):
    answer = "Si è verificato un errore"

    if token == "":
        answer = "Inserisci il token di Hugging Face"
        doc_source = ["nessuna fonte"]
    else:
        # Force the model to answer in Italian and use retrieved context
        italian_prompt = (
            "Rispondi sempre in lingua italiana.\n"
       
            f"Domanda: {question}"
        )

        response = st.session_state.conversation({"question": italian_prompt})
        answer = response.get("answer", "")


        explanation = response.get("source_documents", [])
        doc_source = [d.page_content for d in explanation]

    return answer, doc_source
