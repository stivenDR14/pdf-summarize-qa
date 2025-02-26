import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from utils import TRANSLATIONS
import chromadb
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from transformers import AutoProcessor, AutoModelForImageTextToText
from utils import render_ocr_text

OLLAMA_LLM = "granite3.1-dense"
OLLAMA_EMBEDDINGS = "granite-embedding:278m"

class PDFProcessor:
    def __init__(self):
        self.vectorstore = None
        self.language = "English"
        self.llm = OllamaLLM(model=OLLAMA_LLM)
    
    def set_language(self, language):
        self.language = language
    
    def process_pdf(self, pdf_file, image_file, chunk_size, chunk_overlap):
        
        #if pdf_fil is pdf:
        if pdf_file is not None or image_file is not None:
            print(pdf_file)
            print(image_file)
            if pdf_file is not None:
                loader = PyPDFLoader(file_path=pdf_file.name)
                documents = loader.load()
                print(documents)    
        # Dividir texto
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                print(text_splitter)
                texts = text_splitter.split_documents(documents)
                
                embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDINGS)
                
                new_client = chromadb.EphemeralClient()
                self.vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    client=new_client,
                    collection_name="pdf_collection"
                    #persist_directory="./chroma_db"
                )
            else:
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

                render_ocr_text(res, "output.html", format_text=True)
        
        
        
        return TRANSLATIONS[self.language]["pdf_processed"]

    def get_qa_response(self, message, history):
        if not self.vectorstore:
            return TRANSLATIONS[self.language]["load_pdf_first"]
        
        retriever = self.vectorstore.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        result = qa_chain.invoke({"query": f"{message}.\n answer it in {self.language} "})

        print(result)
        return result["result"] + "\n\n" + "\n\n".join([f"Page: {doc.metadata['page_label']}" for doc in result["source_documents"]])

    def get_summary(self):
        """Generates a summary of the entire document by summarizing each retrieved chunk individually."""
        if not self.vectorstore:
            return TRANSLATIONS[self.language]["load_pdf_first"]
            
        # Get all documents from the vectorstore
        documents = self.vectorstore.get()
        
        # Check if we have documents to summarize
        if not documents or "documents" not in documents or not documents["documents"]:
            return TRANSLATIONS[self.language]["no_documents_to_summarize"]
            
        summaries = []
        
        summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="""
            Summarize the following text concisely while keeping all key points:
            take into account that there are some fragments of text, and some parts of the text at the start or the end will be cutted, and connect the previous and next fragment of text.
            {text}
            """
        )
        
        summary_chain = summary_prompt | self.llm
        
        # Process each document to create individual summaries
        for i, doc_content in enumerate(documents["documents"]):
            print(i)
            if doc_content:  # Check if document content exists
                metadata = documents["metadatas"][i] if "metadatas" in documents and i < len(documents["metadatas"]) else {}
                doc_id = documents["ids"][i] if "ids" in documents and i < len(documents["ids"]) else f"doc_{i}"
                
                chunk_summary = summary_chain.invoke({"text": doc_content})
                summaries.append(f"Source: {doc_id}\nMetadata: {metadata}\nSummary: {chunk_summary}")
        
        print(f"Summaries generated: {len(summaries)}")
        
        # If no summaries were generated, return a message
        if not summaries:
            return TRANSLATIONS[self.language]["no_summaries_generated"]
        
        final_summary_prompt = PromptTemplate(
            input_variables=["summaries", "language"],
            template="""
            Combine the following summaries into a final extended summary:
            {summaries}
            
            The summary must be in {language}.
            """
        )
        
        final_summary_chain = final_summary_prompt | self.llm
        final_summary = final_summary_chain.invoke({"summaries": "\n".join(summaries), "language": self.language})
        
        return final_summary