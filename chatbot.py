import streamlit as st
import os
import falcon
import torch
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from dotenv import load_dotenv


load_dotenv()
# # Funzioni per la gestione della memoria GPU
# def clear_gpu_memory():
#     torch.cuda.empty_cache()
#     gc.collect()

def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"In attesa di {min_memory_available} byte di memoria GPU libera. Riprovo tra {sleep_time} secondi...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Impossibile ottenere {min_memory_available} byte di memoria GPU libera dopo {max_retries} tentativi.")

def display_chatbot_page():
    st.title("Assistente alla Consultazione Documentale del Comune")
    env_token = os.getenv('API_KEY')
    vector_store_list = os.listdir("vector store/")

    # Ensure there's at least one element to choose from
    if len(vector_store_list) > 1:
    # Determine the default choice index
        default_choice = (
            vector_store_list.index('naruto_snake') if 'naruto_snake' in vector_store_list else 0
    )
    
    # Use st.slider to select an index for the vector store
        index = st.slider("", 0, len(vector_store_list) - 1, default_choice)
    
    # Get the selected vector store from the list
        existing_vector_store = vector_store_list[index]

    else:
    # If there's only one item in the list, no need for a slider
        existing_vector_store = vector_store_list[0]
 
    temperature = 0.01
    max_length = 500

    # Prepara il modello LLM se il token è disponibile
    if env_token:
        st.session_state.conversation = falcon.prepare_rag_llm(
            env_token, existing_vector_store, temperature, max_length
        )

    # Inizializza la cronologia della chat
    if "history" not in st.session_state:
        st.session_state.history = []

    # Visualizza i messaggi precedenti
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Inserimento della domanda
    if question := st.chat_input("Fai una domanda"):
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        # Genera la risposta e recupera la fonte
        answer, doc_source = falcon.generate_answer(question, env_token)
        cleaned_doc_source = " ".join([doc.replace("\n", " ") for doc in doc_source])
        with st.chat_message("assistant"):
            #st.markdown(answer)
            st.markdown(f"{cleaned_doc_source}")
        st.session_state.history.append({"role": "assistant", 
                                         "content": f"{doc_source}"})

    # Espandi per visualizzare lo storico della chat e le fonti
    #with st.expander("Storico Chat e Informazioni sulle Fonti"):
        #st.write(st.session_state.history)

def main():
    #clear_gpu_memory()
    display_chatbot_page()

if __name__ == "__main__":
    main()
