from langchain_huggingface import HuggingFaceEmbeddings
from app.common.logger import get_logger
from app.config.config import HUGGINGFACE_TOKEN, HUGGINGFACE_REPO_ID
from app.common.custom_exception import CustomException

logger = get_logger(__name__)

def get_embeddings_model():
    try:
        logger.info("Initializing HuggingFace embeddings model")
        
        model = HuggingFaceEmbeddings(
            model_name= 'sentence-transformers/all-MiniLM-L6-v2',
        )
        
        logger.info('HuggingFace embeddings model loaded successfully')
        
        return model
    
    except Exception as e:
        error_message = CustomException('Failed to load HuggingFace embeddings model', e)
        logger.error(str(error_message))
        raise error_message
    
    
        