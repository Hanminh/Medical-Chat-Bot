from langchain_community.vectorstores import FAISS
from app.components.embeddings import get_embeddings_model
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DB_FAISS_PATH
import os
logger = get_logger(__name__)

def load_vector_store():
    try:
        embeddings_model = get_embeddings_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f'Loading existing FAISS vectorstore ....')        
            return FAISS.load_local(
                DB_FAISS_PATH,
                embeddings_model,
                allow_dangerous_deserialization= True
            )
        else:
            logger.warning("No FAISS vectorstore found at the specified path.")

    except Exception as e:
        error_message = f"Failed to load FAISS vectorstore: {str(e)}"
        logger.error(str(error_message))
        
def save_vector_store(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks provided to save in vector store.")
        
        logger.info("Generateting FAISS vectorstore ....")

        embeddings_model = get_embeddings_model()
        
        db = FAISS.from_documents(text_chunks, embeddings_model)

        logger.info("Saving Faiss vectorstore ....")

        db.save_local(DB_FAISS_PATH)
    
        logger.info("FAISS vectorstore saved successfully.")
        return db
    
    except Exception as e:
        error_message = f"Failed to save FAISS vectorstore: {str(e)}"
        logger.error(str(error_message))
        raise CustomException(error_message) from e
        