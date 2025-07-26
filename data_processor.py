import os
import re
import uuid
import logging
import PyPDF2
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Document
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Enhanced PDF text extraction optimized for Bengali content"""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        
                        if not page_text.strip():
                            continue
                        
                        # Enhanced cleaning for Bengali text
                        cleaned_text = re.sub(
                            r'[^\u0980-\u09FF\u0020-\u007F\u00A0-\u017F\s\.।,!?\d\-:\(\)\"\'\–—""''৳%]',
                            '',
                            page_text
                        )
                        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                        
                        # Remove page numbers and headers/footers
                        lines = cleaned_text.split('\n')
                        meaningful_lines = []
                        
                        for line in lines:
                            line = line.strip()
                            if len(line) > 5:
                                if not re.match(r'^\d+$', line) and not re.match(r'^page\s*\d+', line, re.IGNORECASE):
                                    meaningful_lines.append(line)
                        
                        page_content = '\n'.join(meaningful_lines)
                        
                        if page_content.strip():
                            text += page_content + "\n\n"
                            
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            raise
            
        logger.info(f"Total extracted text length: {len(text)} characters")
        return text.strip()
    
    def preprocess_and_chunk_text(self, text: str) -> List[Document]:
        """Enhanced text preprocessing and chunking with metadata"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
            separators=[
                "\n\n\n",
                "\n\n",
                "\n",
                "।",
                "!",
                "?",
                ".",
                ",",
                ";",
                ":",
                " ",
                ""
            ]
        )
        
        chunks = text_splitter.split_text(text)
        
        # Create LlamaIndex Documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Extract potential chapter/section info
            chapter_match = re.search(r'(অধ্যায়|Chapter|পরিচ্ছেদ)\s*[\d\-]+', chunk, re.IGNORECASE)
            chapter = chapter_match.group(0) if chapter_match else f"Section {i+1}"
            
            # Detect character names for story context
            character_names = []
            bengali_names = re.findall(r'(অনুপম|কল্যাণী|শুম্ভুনাথ|মামা)', chunk)
            character_names.extend(bengali_names)
            
            metadata = {
                "chunk_id": f"chunk_{i}",
                "chunk_index": i,
                "chapter": chapter,
                "characters": list(set(character_names)),
                "chunk_size": len(chunk),
                "language": "bengali" if re.search(r'[\u0980-\u09FF]', chunk) else "english"
            }
            
            documents.append(Document(
                text=chunk,
                metadata=metadata
            ))
        
        logger.info(f"Created {len(documents)} chunks")
        return documents

    def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for a given text"""
        return self.embeddings.embed_query(text)