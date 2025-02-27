import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from ibm_watsonx_ai import APIClient, Credentials
from utils import AI_MODELS, TRANSLATIONS
import chromadb
import requests
import os
OLLAMA_LLM = "granite3.1-dense"
OLLAMA_EMBEDDINGS = "granite-embedding:278m"


endpoint_watsonx = "https://us-south.ml.cloud.ibm.com"

def set_up_watsonx(api_key, project_id_watsonx):
    token_watsonx = authenticate_watsonx(api_key)
    if token_watsonx == None:
        return None
    parameters = {
        "max_new_tokens": 1500,
        "min_new_tokens": 1,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1,
    }

    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    credentials = Credentials(
        url = endpoint_watsonx,
        api_key = api_key,
    )

    client = APIClient(credentials, project_id=project_id_watsonx)

    client.set_token(token_watsonx)

    watsonx_llm = WatsonxLLM(
        model_id="ibm/granite-3-2-8b-instruct",
        watsonx_client=client,
        params = parameters
    )


    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url=endpoint_watsonx,
        project_id=project_id_watsonx,
        params=embed_params,
    ) 

    return watsonx_llm, watsonx_embedding

def authenticate_watsonx(api_key):
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key
    }

    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code == 200:
        token = response.json().get('access_token')
        os.environ["WATSONX_TOKEN"] = token
        return token
    else:
        print("Authentication failed. Status code:", response.status_code)
        print("Response:", response.text)
        return None


class PDFProcessor:
    def __init__(self):
        self.vectorstore = None
        self.language = "English"
    
    def set_language(self, language):
        self.language = language

    def set_llm(self, ai_model, type_model, api_key, project_id_watsonx):
        if ai_model == "Open AI / GPT-4o-mini":
            current_llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.5,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key=api_key, 
            )
            embeding_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                api_key=api_key,
            )


        elif ai_model == "IBM Granite3.1 dense / Ollama local":
            if type_model == "Local":
                current_llm = OllamaLLM(model=OLLAMA_LLM)
                embeding_model = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS)
            else:
                current_llm, embeding_model = set_up_watsonx(api_key, project_id_watsonx)
        else: 
            current_llm = HuggingFaceEndpoint(
                repo_id= AI_MODELS[ai_model],
                temperature=0.5,
            )
            embeding_model = HuggingFaceEmbeddings(
                model_name="ibm-granite/granite-embedding-278m-multilingual",
            )
        return current_llm, embeding_model
    

    def process_pdf(self, pdf_file, chunk_size, chunk_overlap, ai_model, type_model, api_key, project_id_watsonx):
        print(ai_model, type_model, api_key, project_id_watsonx)
        if (ai_model == "Open AI / GPT-4o-mini" and (api_key == "")) or (ai_model == "IBM Granite3.1 dense / Ollama local" and type_model == "Api Key" and (api_key == "" or project_id_watsonx == "")):
            return TRANSLATIONS[self.language]["api_key_required"]
        if pdf_file is not None:
                loader = PyPDFLoader(file_path=pdf_file.name)
                documents = loader.load()
                #delete empty page_content documents from documents
                documents = [doc for doc in documents if doc.page_content]
                print(documents) 
                if(ai_model == "Open AI / GPT-4o-mini" or ai_model == "IBM Granite3.1 dense / Ollama local"):
                    text_splitter = RecursiveCharacterTextSplitter(
                        #is_separator_regex=True,
                        #separators=["\n", "\n\n"],
                        chunk_size=1000,
                        chunk_overlap=100
                    )
                else: 
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                #print(text_splitter)
                texts = text_splitter.split_documents(documents)
                print(len(texts))
                _, embeddings = self.set_llm(ai_model, type_model, api_key, project_id_watsonx)
                
                new_client = chromadb.EphemeralClient()
                
                self.vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    client=new_client,
                    collection_name="pdf_collection"
                    #persist_directory="./chroma_db"
                )
                
                return TRANSLATIONS[self.language]["pdf_processed"]
                """ else:
                device = "cpu"
                model = AutoModelForImageTextToText.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf", device_map=device)
                processor = AutoProcessor.from_pretrained("stepfun-ai/GOT-OCR-2.0-hf")

                inputs = processor(image_file.name, return_tensors="pt", format=True).to(device)

                generate_ids = model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=processor.tokenizer,
                    stop_strings="<|im_end|>",
                    max_new_tokens=4096,
                    repetition_penalty=1.5,        # Penalize repeated tokens
                    no_repeat_ngram_size=3,        # Prevent repeating of 3-grams
                    early_stopping=True,           # Stop generation when criteria met
                    num_beams=4,                   # Use beam search for better results
                    length_penalty=1.0, 
                )

                res=processor.decode(generate_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                print(res)

                render_ocr_text(res, "output.html", format_text=True) """
        
        else:
            return TRANSLATIONS[self.language]["load_pdf_first"]
        

    def get_qa_response(self, message, history, ai_model, type_model, api_key, project_id_watsonx):
        current_llm, _ = self.set_llm(ai_model, type_model, api_key, project_id_watsonx)

        if not self.vectorstore:
            return TRANSLATIONS[self.language]["load_pdf_first"]
        
        retriever = self.vectorstore.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=current_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": f"{message}.\n answer it in {self.language}. Remember not to mention anything that is not in the text. Do not extend information that is not provided in the text. "})

        unique_page_labels = {doc.metadata['page_label'] for doc in result["source_documents"]}
        
        page_labels_text = "\n".join([f"Page: {page}" for page in sorted(unique_page_labels)])

        return result["result"] + "\n\nSources: " + page_labels_text

    def get_summary(self, ai_model, type_model, api_key, project_id_watsonx):
        """Generates a summary of the entire document by summarizing each retrieved chunk individually."""
        if not self.vectorstore:
            return TRANSLATIONS[self.language]["load_pdf_first"]
            
        # Get all documents from the vectorstore
        documents = self.vectorstore.get()
        
        current_llm, _ = self.set_llm(ai_model, type_model, api_key, project_id_watsonx)
        # Check if we have documents to summarize
        if not documents or "documents" not in documents or not documents["documents"]:
            return TRANSLATIONS[self.language]["no_documents_to_summarize"]
            
        summaries = []
        
        summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Extract all key points from the following text:
            {text}

            Remember not to mention anything that is not in the text.
            Do not extend information that is not provided in the text.
            The summary must be in {language}.
            """
        )
        
        summary_chain = summary_prompt | current_llm
        
        # Process each document to create individual summaries
        for i, doc_content in enumerate(documents["documents"]):
            print(i)
            if doc_content:  # Check if document content exists
                #metadata = documents["metadatas"][i] if "metadatas" in documents and i < len(documents["metadatas"]) else {}
                #doc_id = documents["ids"][i] if "ids" in documents and i < len(documents["ids"]) else f"doc_{i}"
                
                chunk_summary = summary_chain.invoke({"text": doc_content,"language": self.language})
                summaries.append(f"Summary fragment{i}: {chunk_summary}")
                print(f"fragment{i}: {doc_content}")
        
        print(f"Summaries generated: {len(summaries)}")
        
        # If no summaries were generated, return a message
        if not summaries:
            return TRANSLATIONS[self.language]["no_summaries_generated"]
        
        final_summary_prompt = PromptTemplate(
            input_variables=["summaries", "language"],
            template="""
            Combine the following summaries into a final concise summary of between 2 and 4 paragraphs:
            {summaries}

            Remember not to mention anything that is not in the text.
            Do not extend information that is not provided in the text.
            The summary must be in {language}.
            """
        )
        
        final_summary_chain = final_summary_prompt | current_llm
        final_summary = final_summary_chain.invoke({"summaries": "\n".join(summaries), "language": self.language})
        
        return final_summary