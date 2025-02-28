import json
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
from dotenv import load_dotenv

OLLAMA_LLM = "granite3.1-dense"
OLLAMA_EMBEDDINGS = "granite-embedding:278m"


load_dotenv()

api_key_watsonx = os.getenv('WATSONX_APIKEY')
projectid_watsonx = os.getenv('WATSONX_PROJECT_ID')
endpoint_watsonx = "https://us-south.ml.cloud.ibm.com"

def set_up_watsonx():
    token_watsonx = authenticate_watsonx(api_key_watsonx)
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
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 1,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
    }

    credentials = Credentials(
        url = endpoint_watsonx,
        api_key = api_key_watsonx,
    )

    client = APIClient(credentials, project_id=projectid_watsonx)

    client.set_token(token_watsonx)

    watsonx_llm = WatsonxLLM(
        model_id="ibm/granite-3-2-8b-instruct",
        watsonx_client=client,
        params = parameters
    )


    watsonx_embedding = WatsonxEmbeddings(
        model_id="ibm/granite-embedding-278m-multilingual",
        url=endpoint_watsonx,
        project_id=projectid_watsonx,
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
                try:
                    # Verificar que Ollama está funcionando y el modelo está disponible
                    current_llm = OllamaLLM(model=OLLAMA_LLM)
                    # Intenta hacer un embedding de prueba
                    test_embedding = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS)
                    test_embedding.embed_query("test")
                    embeding_model = test_embedding
                except Exception as e:
                    print(f"Error with Ollama: {e}")
                    # Fallback a otro modelo o manejo de error
                    raise Exception("Please ensure Ollama is running and the models are pulled: \n" +
                                  f"ollama pull {OLLAMA_LLM}\n" +
                                  f"ollama pull {OLLAMA_EMBEDDINGS}")
            else:
                current_llm, embeding_model = set_up_watsonx()
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
        defined_chunk_size = 1000
        defined_chunk_overlap = 100
        if (ai_model == "Open AI / GPT-4o-mini" and (api_key == "")) : #or (ai_model == "IBM Granite3.1 dense / Ollama local" and type_model == "Api Key" and (api_key == "" or project_id_watsonx == "")
            return TRANSLATIONS[self.language]["api_key_required"]
        if pdf_file is not None:
                loader = PyPDFLoader(file_path=pdf_file.name)
                documents = loader.load()
                #delete empty page_content documents from documents
                documents = [doc for doc in documents if doc.page_content]
                if(ai_model == "Open AI / GPT-4o-mini" or ai_model == "IBM Granite3.1 dense / Ollama local"):
                    if type_model == "Api Key":
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=defined_chunk_size,
                            chunk_overlap=defined_chunk_overlap,
                            separators=["\n\n", "\n"] 
                        )
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=defined_chunk_size,
                            chunk_overlap=defined_chunk_overlap,
                        )
                else: 
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=defined_chunk_size,
                        chunk_overlap=defined_chunk_overlap
                    )

                #print(text_splitter)
                texts = text_splitter.split_documents(documents)
                _, embeddings = self.set_llm(ai_model, type_model, api_key, project_id_watsonx)

                #delete all documents from the vectorstore
                if self.vectorstore:
                    self.vectorstore.delete_collection()
                
                new_client = chromadb.EphemeralClient()
                
                self.vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    client=new_client,
                    collection_name="pdf_collection"
                    #persist_directory="./chroma_db"
                )
                
                return TRANSLATIONS[self.language]["pdf_processed"] + f" ---- Chunks: {len(self.vectorstore.get()["documents"])}"
        
        else:
            return TRANSLATIONS[self.language]["load_pdf_first"]
        

    def get_qa_response(self, message, history, ai_model, type_model, api_key, project_id_watsonx, k=4):
        current_llm, _ = self.set_llm(ai_model, type_model, api_key, project_id_watsonx)

        if not self.vectorstore:
            return TRANSLATIONS[self.language]["load_pdf_first"]
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

        qa_chain = RetrievalQA.from_chain_type(
            llm=current_llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )
        
        result = qa_chain.invoke({"query": f"{message}.\n You must answer it in {self.language}. Remember not to mention anything that is not in the text. Do not extend information that is not provided in the text. "})

        unique_page_labels = {doc.metadata['page_label'] for doc in result["source_documents"]}
        
        page_labels_text = " & ".join([f"Page: {page}" for page in sorted(unique_page_labels)])

        return result["result"] + "\n\nSources: " + page_labels_text
        

    def summarizer_by_k_top_n(self, ai_model, type_model, api_key, project_id_watsonx, k, summary_prompt, just_get_documments=False):
        if not self.vectorstore:
            return TRANSLATIONS[self.language]["load_pdf_first"]
        
        current_llm, _ = self.set_llm(ai_model, type_model, api_key, project_id_watsonx)
        # Get all documents from the vectorstore
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        documents = retriever.invoke('Summary of the document and key points')

        if just_get_documments:
            return  "\n".join([doc.page_content for doc in documents])
        
        summary_chain = summary_prompt | current_llm
        final_summary = summary_chain.invoke({"texts": "\n".join([doc.page_content for doc in documents]), "language": self.language})
        return final_summary

        # Get the top k documents by score
    def get_summary(self, ai_model, type_model, api_key, project_id_watsonx, just_get_documments=False, k=10):

        final_summary_prompt = PromptTemplate(
            input_variables=["texts", "language"],
            template="""
            Combine the following texts into a cohesive and structured final summary:   
            ------------
            {texts}
            ------------
            The final summary should be between 2 and 4 paragraphs.
            Preserve the original meaning without adding external information or interpretations.
            Ensure clarity, logical flow, and coherence between the combined points.
            The summary must be in {language}.
            """
        )
        
        return self.summarizer_by_k_top_n(ai_model, type_model, api_key, project_id_watsonx, k, final_summary_prompt, just_get_documments)
    
    

    def get_specialist_opinion(self, ai_model, type_model, api_key, project_id_watsonx, specialist_prompt):
        questions_prompt = PromptTemplate(
            input_variables=["text", "specialist_prompt", "language"],
            template="""
            * Act as a specialist based on the following instructions and behaviour that you will follow:
            ------------
            {specialist_prompt}
            ------------
            * Based on your role as specialist, create some different sintetized and concise aspects to ask to the knowledge base of the document about the following text:
            ------------
            {text}
            ------------
            * The key aspects and questions must be provided in JSON format with the following structure:
            {{
                "aspects": [
                    "Aspect 1",
                    "Aspect 2",
                    "Aspect 3",
                    "Aspect 4",
                    "Aspect 5",
                    "Aspect 6",
                    "Aspect 7",
                    "Aspect 8",
                    "Aspect 9",
                    "Aspect 10",
                ]
            }}
            ------------
            *Example of valid output:
            {{
                "aspects": [
                    "Finished date of the project",
                    "Payment of the project",
                    "Project extension"
                    ]
            }}
            ------------
            * The aspects must be redacted in the language of {language}.
            * The given structure must be followed strictly in front of the keys, just use the list of aspects, do not add any other key.
            * Generate until 10 different aspects.
            ------------
            Answer: 
            """
        )
        if not self.vectorstore:
            return TRANSLATIONS[self.language]["load_pdf_first"]
        
        current_llm, _ = self.set_llm(ai_model, type_model, api_key, project_id_watsonx)

        summary_text = self.get_summary(ai_model, type_model, api_key, project_id_watsonx, True, 10)
        questions_chain = questions_prompt | current_llm
        questions = questions_chain.invoke({"text": summary_text, "specialist_prompt": specialist_prompt, "language": self.language})

        print(questions)

        #clean the questions variable, delete all the text before the json and after the json
        questions = questions.split("{")[1]
        questions = questions.split("}")[0]
        questions = questions.strip()
        print(questions)
        questions = json.loads(questions)

        print(questions)

        if len(questions["aspects"]) > 15:
            questions["aspects"] = questions["aspects"][:15]
        else:
            questions["aspects"] = questions["aspects"]

        aspects_text = "\n".join([f"* {aspect}: {self.get_qa_response(aspect, [], ai_model, type_model, api_key, project_id_watsonx, 2)}" for aspect in questions["aspects"]])

        return aspects_text
    
    
    """ Actúa como un abogado altamente experimentado en derecho civil y contractual.

    Examina si existen cláusulas abusivas, desproporcionadas o contrarias a la normativa vigente, y explícalas con claridad.
    Basa tu análisis en principios relevantes del derecho civil y contractual.
    Ofrece un argumento estructurado y recomendaciones prácticas.
    Si hay múltiples interpretaciones posibles, preséntalas de manera objetiva.
    Mantén un tono profesional, preciso y fundamentado.

    Basado en lo que analices, proporciona una evaluación legal detallada """

    """ Actúa como un asesor e ingeniero financiero experto en lectura de reportes y análisis de datos.
    
    Basado en los datos y conclusiones del reporte, proporciona una evaluación financiera detallada y posibles escenarios tanto negativos como positivos que se puedan presentar.
    Establece el riesgo que se corre en cada escenario, la probabilidad de ocurrencia de cada uno y la magnitud del impacto en el recurso.
    Si hay múltiples interpretaciones posibles, preséntalas de manera objetiva.
    Realiza una hipótesis que pronostique el futuro de la situación o recurso analizado, teniendo en cuenta los datos y conclusiones del reporte.
    Presenta tus hipotesis en 3 aspectos, corto, mediano y largo plazo.
    Mantén un tono profesional, preciso y fundamentado.
    
    Basado en lo que analices, proporciona una evaluación en detalle sobre los activos, reportes y/o recursos que se analizaron"""