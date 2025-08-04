from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from app.components.vector_store import load_vector_store
from app.components.llm import load_llm

from app.config.config import HUGGINGFACE_REPO_ID, HUGGINGFACE_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
import os


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """ Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context: {context}

Question: {question}

Answer:

"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
def create_qa_chain():
    try:
        logger.info("Loading vector store ...")
        db = load_vector_store()
        
        if db is None:
            raise CustomException("Vector store is not available. Please ensure it is created before running the QA chain.")
        
        llm = load_llm(
            huggingface_repo_id=HUGGINGFACE_REPO_ID,
            hf_token=HUGGINGFACE_TOKEN
        )
        
        if llm is None:
            raise CustomException("LLM is not available. Please ensure it is loaded correctly before running the QA chain.")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm= llm,
            chain_type= 'stuff',
            retriever=db.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": set_custom_prompt()
            }
        )
        logger.info("QA chain created successfully.")
        return qa_chain
    
    except Exception as e:
        error_message = f"Failed to create QA chain: {str(e)}"
        logger.error(str(error_message))
        raise CustomException(error_message) from e
        
        

