
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

          #creation des dictionnaires
        composante_description = {"Name" : 'Description', "Type" : "0", "Value" : ""}
        composante_name = {"Name" : 'Name', "Type" : "0", "Value" : ""}
        operation_temperature = {"Name" : "Operation Temperature", "Type" : '4', "Minimum": "", "Maximum" : "" , "Units" : ""}
        composante_weight = {"Name" : 'Weight', "Type" : 4, "Value" :"","Units": ""}
        composant_direction={"Name" : 'Direction', "Type" : 8, "Value" : ""}
        composant_output_current = {"Name" : 'Output current', "Type" : 4, "Value" : "","Units": ""}
    
        query = "only the first name of this model  "
        docs = docsearch.similarity_search(query)
        manufacturer_reference = chain.run(input_documents=docs, question=query)
        
        
        query = "Give me only the number of weight of this model without unit "
        docs = docsearch.similarity_search(query)
        weight = float(chain.run(input_documents=docs, question=query))
        composante_weight["Value"]=weight
        
        query = "Give me only the unit of weight "
        docs = docsearch.similarity_search(query)
        weight_unit = chain.run(input_documents=docs, question=query)
        composante_weight["Units"]=weight_unit
        
        query = "Give me the direction of this model, only is it 'output' or 'input' "
        docs = docsearch.similarity_search(query)
        direction = chain.run(input_documents=docs, question=query)
        composant_direction["Value"]=direction
        
        query = "Give me only the number of the output current without unit"
        docs = docsearch.similarity_search(query)
        output_current = float(chain.run(input_documents=docs, question=query))
        composant_output_current["Value"]= output_current
        
        query = "Give me only the symbole unit of the output current"
        docs = docsearch.similarity_search(query)
        output_current_unit = chain.run(input_documents=docs, question=query)
        composant_output_current["Units"]=output_current_unit
        
        query = "give me only the name of this model ?"
        docs = docsearch.similarity_search(query)
        model_name = chain.run(input_documents=docs, question=query)
        
        composante_name['Value'] = model_name
        
        
        query = "give me a description of this model"
        docs = docsearch.similarity_search(query)
        model_description = chain.run(input_documents=docs, question=query)
        
        composante_description['Value'] = model_description
        
        
        
        query = "Give me  the integer response of the minimum operation temperature without including the unit symbole "
        docs = docsearch.similarity_search(query)
        minimum = float(chain.run(input_documents=docs, question=query))
        
        query = "Give me  the integer response of the maximum operation temperature without including the unit symbole"
        docs = docsearch.similarity_search(query)
        maximum = float(chain.run(input_documents=docs, question=query))
        
        query = "Give me only the symbole of the unit of the operation temperature"
        docs = docsearch.similarity_search(query)
        unit = chain.run(input_documents=docs, question=query)
        
        operation_temperature["Maximum"] = maximum
        operation_temperature["Minimum"] = minimum
        operation_temperature["Units"] = unit


    
    
    
        # use for debugging.
        import logging
    
        logging.getLogger().setLevel(logging.CRITICAL)
    
    
    
        # Define a directory path for saving JSON files
        json_dir = "json_files"  # Change this path to your desired directory on Google Colab
    
        # Ensure the directory exists
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
    
    
        with st.form(key='form1'):
            col1 , col2 = st.columns([2,1])
            with col1:
                placeholder1 = st.text_input("Name", value=composante_name['Value'])
                
                categorie = st.text_input("Manufacturer reference", value = manufacturer_reference)
                placeholder2 = st.text_area("Description", value= composante_description["Value"])  

      

            with col2:
              image = Image.open('exemple_image/LC.jpg')
              st.image(image, caption='LC')
              # Important

            submit_button_1 = st.form_submit_button(label='Save_1')
            if submit_button_1:
              composante_name['Value'] = placeholder1
              composante_description['Value'] = placeholder2
    
        with st.form(key='form2'):
            col1,col2,col3 = st.columns([3,2,1])
            with col1:
              st.markdown('<span style="color: orange; font-weight: bold;">Temperature</span>', unsafe_allow_html=True)
              placeholder3 = st.number_input(':blue[Maximum operation Temperature]', value=operation_temperature["Maximum"])
              placeholder4 = st.number_input(':blue[Minimum operation Temperature]', value=operation_temperature['Minimum'])
              placeholder5 = st.text_input(':blue[Units]', value=operation_temperature['Units'])

            with col3:
              st.markdown('<span style="color: orange; font-weight: bold;">Weight</span>', unsafe_allow_html=True)
              placeholder6 = st.number_input(':blue[Weight]', value=composante_weight["Value"])
              placeholder7 = st.text_input(':blue[Weight Units]', value=composante_weight['Units'])
        
            with col2:
              st.markdown('<span style="color: orange; font-weight: bold;">Current</span>', unsafe_allow_html=True)
              placeholder9 = st.number_input(':blue[Output current]', value=composant_output_current["Value"])
              placeholder10 = st.text_input(':blue[Output current Units]', value=composant_output_current['Units'])
              placeholder8 = st.text_input(':blue[Direction]', value=composant_direction["Value"])

            submit_button_2 = st.form_submit_button(label='Save_2')
            if submit_button_2:
              operation_temperature['Maximum']= placeholder3
              operation_temperature['Minimum'] = placeholder4
              operation_temperature['Units'] = placeholder5
              composante_weight["Value"]=placeholder6
              composante_weight['Units']=placeholder7
              composant_direction["Value"]=placeholder8
              composant_output_current["Value"]=placeholder9
              composant_output_current['Units']=placeholder10
    
        # Download Button
        if st.button("Download JSON"):
            data = {
                "Type":"Element/System/Component",
                "Parameters":[
                    composante_name,
                    composante_description,
                    operation_temperature,
                    composante_weight,
                    composant_direction,
                    composant_output_current
                    ]
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
    

            
