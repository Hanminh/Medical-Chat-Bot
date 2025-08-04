import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from app.config.config import DATA_PATH, CHUNK_OVERLAP, CHUNK_SIZE

logger = get_logger(__name__)

def load_pdf_file():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path does not exits")

        logger.info("Loading PDF files from data directory")
        loader = DirectoryLoader(
            DATA_PATH,
            glob= "*.pdf",
            loader_cls= PyPDFLoader
        )
        documents = loader.load()
        
        if not documents:
            logger.warning("No PDF files found in the specified directory.")
        else:
            logger.info(f"Loaded {len(documents)} PDF files successfully.")
            
        return documents
    
    except Exception as e:
        error_message = CustomException('Failed to load PDF files', e)
        logger.error(str(error_message))
        return []
    
def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents to process for text chunking")

        logger.info(f'Splitting {len(documents)} documents into chunks')
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size= CHUNK_SIZE,
            chunk_overlap= CHUNK_OVERLAP,
        )
        text_chunks = text_splitter.split_documents(documents)
        return text_chunks

    except Exception as e:
        error_message = CustomException('Failed to create text chunks', e)
        logger.error(str(error_message))
        return []