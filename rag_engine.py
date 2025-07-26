import os
import uuid
import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from llama_index.core import VectorStoreIndex, ServiceContext, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from sklearn.metrics.pairwise import cosine_similarity

from data_processor import DataProcessor

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.translator_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.data_processor = DataProcessor()
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "bengali-rag-index"
        self.dimension = 1536
        
        # Create index if it doesn't exist
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        # Build vector store and index
        self.vector_store = PineconeVectorStore(
            pinecone_index=self.pc.Index(self.index_name),
            add_sparse_vector=True,
        )
        
        # Initialize service context for LlamaIndex
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.data_processor.embeddings,
            node_parser=SentenceSplitter(chunk_size=600, chunk_overlap=100)
        )
        
        # Load the index if exists, otherwise it will be built later
        self.vector_index = None
        
        # Reranker
        self.reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=5
        )
        
        # Memory for conversations
        self.conversations = {}
        
        # Agent setup
        self.agent = self._create_agent()
        
        logger.info("RAG System initialized successfully")
    
    def build_index(self, pdf_path: str):
        """Build and populate the vector index from a PDF"""
        logger.info("Building vector index...")
        text = self.data_processor.extract_pdf_text(pdf_path)
        documents = self.data_processor.preprocess_and_chunk_text(text)
        
        # Create vector index
        self.vector_index = VectorStoreIndex.from_documents(
            documents, 
            service_context=self.service_context,
            vector_store=self.vector_store
        )
        logger.info("Vector index built successfully")
    
    def detect_language(self, text: str) -> str:
        """Simple language detection for Bengali vs English"""
        if re.search(r'[\u0980-\u09FF]', text):
            return 'bengali'
        return 'english'
    
    def translate_text(self, text: str, source: str, target: str) -> str:
        """Translate text between English and Bengali using LLM"""
        if source == target:
            return text
        
        response = self.translator_llm.invoke([
            SystemMessage(content=f"You are a professional translator. Translate the following {source} text to {target}."),
            HumanMessage(content=text)
        ])
        
        return response.content
    
    def create_retriever(self, top_k: int = 10) -> VectorIndexRetriever:
        """Create a retriever from the vector index"""
        if self.vector_index is None:
            # Try to load existing index
            self.vector_index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                service_context=self.service_context
            )
        return VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=top_k,
            service_context=self.service_context
        )
    
    def create_query_engine(self) -> RetrieverQueryEngine:
        """Create a query engine with retriever and reranker"""
        retriever = self.create_retriever(top_k=10)
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            service_context=self.service_context
        )
        return RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[self.reranker]
        )
    
    def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query"""
        query_engine = self.create_query_engine()
        response = query_engine.query(query)
        return [{
            'text': node.text,
            'score': node.score,
            'metadata': node.metadata
        } for node in response.source_nodes]
    
    def generate_answer(self, query: str, context: List[Dict[str, Any]], 
                       conversation_history: List[BaseMessage] = None,
                       target_lang: str = 'bengali') -> Dict[str, Any]:
        """Generate answer using retrieved context and conversation history"""
        # Prepare context string
        context_str = "\n\n".join([
            f"[Source {i+1}] {doc['text']}" 
            for i, doc in enumerate(context)
        ])
        
        # Create prompt based on language
        if target_lang == 'bengali':
            system_content = """আপনি একজন সাহায্যকারী সহকারী যিনি বাংলা সাহিত্য, বিশেষ করে এইচএসসি বাংলা ১ম পত্র বই সম্পর্কিত প্রশ্নের উত্তর দেন। নিচের প্রসঙ্গ ব্যবহার করে প্রশ্নের উত্তর দিন। যদি প্রসঙ্গে উত্তর না থাকে, বলুন 'উত্তরটি প্রসঙ্গে নেই'।"""
            human_template = "প্রসঙ্গ:\n{context}\n\nপ্রশ্ন: {question}\nউত্তর:"
        else:
            system_content = """You are a helpful assistant that answers questions about Bengali literature, specifically from the HSC Bangla 1st paper textbook. Use the context below to answer the question. If the context doesn't contain the answer, say 'I don't know'."""
            human_template = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_content),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content=human_template.format(context=context_str, question=query))
        ])
        
        # Prepare messages
        messages = [prompt.messages[0]]
        if conversation_history:
            messages.extend(conversation_history[-4:])  
        messages.append(prompt.messages[-1])
        
        # Generate response
        response = self.llm.invoke(messages)
        
        # Calculate confidence based on retrieval scores
        avg_score = np.mean([doc['score'] for doc in context]) if context else 0.0
        confidence = min(avg_score * 1.2, 1.0) if avg_score > 0 else 0.0
        
        return {
            "answer": response.content,
            "confidence": confidence
        }
    
    def get_or_create_conversation(self, conversation_id: str = None) -> Tuple[str, List[BaseMessage]]:
        """Get existing or create new conversation"""
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "history": [],
                "created_at": datetime.now()
            }
        
        return conversation_id, self.conversations[conversation_id]["history"]
    
    def _create_agent_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        return [
            Tool(
                name="RetrieveDocuments",
                func=self.retrieve_documents,
                description="Retrieve relevant document chunks from the knowledge base"
            ),
            Tool(
                name="GenerateAnswer",
                func=self.generate_answer,
                description="Generate an answer based on context and conversation history"
            )
        ]
    
    def _create_agent(self) -> AgentExecutor:
        """Create an agent for handling RAG queries with translation"""
        tools = self._create_agent_tools()
        
        # Agent prompt template
        system_message = SystemMessage(content=(
            "You are a bilingual assistant that answers questions about Bengali literature. "
            "Follow these steps for every query:\n"
            "1. Detect the language of the query\n"
            "2. For English queries, translate to Bengali for retrieval\n"
            "3. Retrieve relevant context using the translated query\n"
            "4. Generate an answer in the original query language\n"
            "5. For Bengali queries, use the original query for retrieval\n"
            "6. Always maintain context of the conversation history"
        ))
        
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            HumanMessage(content="{input}")
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=4,
                return_messages=True
            )
        )
    
    def agent_query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Query the RAG system using the agent"""
        # Get or create conversation
        conversation_id, history = self.get_or_create_conversation(conversation_id)
        
        # Detect language
        query_lang = self.detect_language(query)
        
        # Prepare agent input
        agent_input = {
            "input": query,
            "chat_history": history
        }
        
        # Execute the agent
        result = self.agent.invoke(agent_input)
        
        # Update conversation history
        self.conversations[conversation_id]["history"].extend([
            HumanMessage(content=query),
            HumanMessage(content=result["output"])
        ])
        
        # Return formatted result
        return {
            "answer": result["output"],
            "sources": [],  
            "conversation_id": conversation_id,
            "confidence_score": 0.9  
        }
    
    def direct_query(self, query: str, conversation_id: str = None) -> Dict[str, Any]:
        """Direct query without agent (for evaluation and simpler queries)"""
        # Get or create conversation
        conversation_id, history = self.get_or_create_conversation(conversation_id)
        
        # Detect query language
        query_lang = self.detect_language(query)
        
        # For English queries: Translate → Retrieve → Answer in English
        if query_lang == 'english':
            # Translate to Bengali for retrieval
            translated_query = self.translate_text(query, 'english', 'bengali')
            # Retrieve with translated query
            context = self.retrieve_documents(translated_query)
            # Generate answer in English
            result = self.generate_answer(query, context, history, target_lang='english')
        else:
            # For Bengali queries: Direct retrieval → Answer in Bengali
            context = self.retrieve_documents(query)
            result = self.generate_answer(query, context, history, target_lang='bengali')
        
        # Update conversation history
        self.conversations[conversation_id]["history"].extend([
            HumanMessage(content=query),
            HumanMessage(content=result["answer"])
        ])
        
        return {
            "answer": result["answer"],
            "sources": context,
            "conversation_id": conversation_id,
            "confidence_score": result["confidence"]
        }
    
    def evaluate_response(self, query: str, expected_answer: str) -> Dict[str, Any]:
        """Evaluate RAG system performance"""
        # Use direct query for evaluation
        response = self.direct_query(query, use_memory=False)
        generated_answer = response["answer"]
        context = response["sources"]
        
        # Calculate evaluation metrics
        # Groundedness: How well the answer is supported by context
        context_embeddings = [self.data_processor.generate_embeddings(doc['text']) for doc in context]
        answer_embedding = self.data_processor.generate_embeddings(generated_answer)
        max_similarity = max(cosine_similarity([answer_embedding], [ctx])[0][0] for ctx in context_embeddings)
        groundedness_score = max(0.0, min(1.0, max_similarity * 1.5))
        
        # Relevance: Quality of retrieval
        relevance_score = np.mean([doc['score'] for doc in context]) if context else 0.0
        
        # similarity: How close to expected answer
        expected_embedding = self.data_processor.generate_embeddings(expected_answer)
        answer_similarity = cosine_similarity([expected_embedding], [answer_embedding])[0][0]
        
        return {
            "query": query,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "groundedness_score": groundedness_score,
            "relevance_score": relevance_score,
            "answer_similarity": answer_similarity,
            "retrieved_contexts": [doc['text'][:200] + "..." for doc in context]
        }