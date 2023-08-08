
import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import json
import os
import requests
import base64

st.title("OpenAI Application")

# Demander à l'utilisateur d'entrer sa clé API OpenAI
openai_api_key = st.text_input("Enter your OpenAI API Key:")

# Si l'utilisateur a soumis une clé API
if openai_api_key:
    with st.form("upload-form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False,
                                        type=['pdf'],
                                        help="Upload a file to annotate")
        submitted = st.form_submit_button("Upload")

    if uploaded_file is not None:
        os.environ["OPENAI_API_KEY"] = openai_api_key

        reader = PdfReader(uploaded_file)

        raw_text = ''
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                raw_text += text
    
        # We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.
    
        text_splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 500,
            chunk_overlap  = 100,
            length_function = len,
            )
        texts = text_splitter.split_text(raw_text)
    
        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
    
        #manufacturer_reference={"name":''}
    
        query = "Give me only the name of the manufacturer reference  "
        docs = docsearch.similarity_search(query)
        manufacturer_reference = chain.run(input_documents=docs, question=query)
    
        composante_name = {"name" : 'Name', "type" : "0", "value" : ""}
    
    
        query = "give me only the name of this model ?"
        docs = docsearch.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
    
        composante_name['value'] = response
        composante_description = {"name" : 'Description', "type" : "0", "value" : ""}
    
        query = "give me a description of this model"
        docs = docsearch.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
    
        composante_description['value'] = response
    
        operation_temperature = {"name" : "Operation Temperature", "type" : '4', "minimum": "", "maximum" : "" , "unit" : ""}
    
        query = "Give me  the integer response of the minimum operation temperature without including the unit symbole "
        docs = docsearch.similarity_search(query)
        minimum = int(chain.run(input_documents=docs, question=query))
    
        query = "Give me  the integer response of the maximum operation temperature without including the unit symbole"
        docs = docsearch.similarity_search(query)
        maximum = int(chain.run(input_documents=docs, question=query))
    
        query = "Give me only the symbole of the unit of the operation temperature"
        docs = docsearch.similarity_search(query)
        unit = chain.run(input_documents=docs, question=query)
    
        operation_temperature["maximum"] = maximum
        operation_temperature["minimum"] = minimum
        operation_temperature["unit"] = unit
    
    
    
    
        # use for debugging.
        import logging
    
        logging.getLogger().setLevel(logging.CRITICAL)
    
    
    
        # Define a directory path for saving JSON files
        json_dir = "json_files"  # Change this path to your desired directory on Google Colab
    
        # Ensure the directory exists
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
    
    
        c1, c2 = st.columns([1, 4])
        c2.subheader("Parameters")
    
        with c1:
            placeholder1 = c2.text_input("Name", value=composante_name['value'])
            placeholder2 = c2.text_area("Description", value= composante_description['value'])
            placeholder3 = c2.number_input("Maximum Temperature", value=operation_temperature["maximum"])
            placeholder4 = c2.text_input("Minimum Temperature", value=operation_temperature['minimum'])
            placeholder5 = c2.text_input("Temperature Unit", value=operation_temperature['unit'])
    
        # Save Button
        if c2.button("Save"):
            composante_name['value'] = placeholder1
            composante_description['value'] = placeholder2
            operation_temperature['maximum']=placeholder3
            operation_temperature['minimum'] = placeholder4
            operation_temperature['unit'] = placeholder5
    
        # Download Button
        if c2.button("Download JSON"):
            data = {
                "composante_name": composante_name,
                "composante_description": composante_description,
                "operation_temperature": operation_temperature
                }
    
            json_data = json.dumps(data, indent=4)
            #file_name = f"{composante_name['value']}_data.json"
            file_name = f"{manufacturer_reference}_data.json"
            file_path = os.path.join(json_dir, file_name)
    
            with open(file_path, 'w') as json_file:
                json_file.write(json_data)
    
            st.success(f"JSON file saved as: {file_path}")
    
    
            def get_download_link(file_path):
                with open(file_path, 'rb') as f:
                    data = f.read()
                b64 = base64.b64encode(data).decode('utf-8')
                href = f'<a href="data:application/json;base64,{b64}" download> Télécharger le fichier json </a>'
                return href
    
            #get_download_link(file_path)
    
            #file_path = os.path.join('/content/json_files', file_name)
            st.markdown(get_download_link(file_path), unsafe_allow_html=True)
    

            
