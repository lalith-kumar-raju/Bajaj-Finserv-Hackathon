from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any, Optional
from models import DocumentChunk
from config import Config
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.config = Config()
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.pinecone = None
        self.index = None
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and index using new API"""
        try:
            # Validate API key
            if not self.config.PINECONE_API_KEY or self.config.PINECONE_API_KEY == "your_pinecone_api_key_here":
                raise Exception("Pinecone API key not configured. Please set PINECONE_API_KEY environment variable.")
            
            # Initialize Pinecone with new API
            self.pinecone = Pinecone(api_key=self.config.PINECONE_API_KEY)
            
            # Check if index exists, create if not
            existing_indexes = self.pinecone.list_indexes().names()
            logger.info(f"Available Pinecone indexes: {existing_indexes}")
            
            if self.config.PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.config.PINECONE_INDEX_NAME}")
                self.pinecone.create_index(
                    name=self.config.PINECONE_INDEX_NAME,
                    dimension=384,  # Dimension for all-MiniLM-L6-v2
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",  # Changed from gcp to aws
                        region="us-east-1"  # Changed to us-east-1
                    )
                )
                logger.info(f"✅ Successfully created Pinecone index: {self.config.PINECONE_INDEX_NAME}")
            else:
                logger.info(f"✅ Using existing Pinecone index: {self.config.PINECONE_INDEX_NAME}")
            
            self.index = self.pinecone.Index(self.config.PINECONE_INDEX_NAME)
            logger.info("✅ Pinecone initialized successfully with new API")
            
            # Test the connection
            stats = self.index.describe_index_stats()
            logger.info(f"✅ Pinecone connection verified. Index stats: {stats.total_vector_count} vectors")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Pinecone: {e}")
            logger.error("Please check your Pinecone configuration:")
            logger.error("1. PINECONE_API_KEY environment variable")
            logger.error("2. PINECONE_ENVIRONMENT environment variable") 
            logger.error("3. PINECONE_INDEX_NAME environment variable")
            logger.error("4. Internet connection and Pinecone service availability")
            raise Exception(f"Pinecone initialization failed: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def store_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Store document chunks in vector database"""
        try:
            logger.info(f"🚀 Starting to store {len(chunks)} chunks in Pinecone...")
            
            # Generate embeddings for chunks
            texts = [chunk.content for chunk in chunks]
            logger.info(f"📊 Generating embeddings for {len(texts)} texts...")
            embeddings = self.generate_embeddings(texts)
            logger.info(f"✅ Generated {len(embeddings)} embeddings successfully")
            
            # Prepare vectors for Pinecone
            vectors = []
            for i, chunk in enumerate(chunks):
                # Clean metadata to ensure all values are valid for Pinecone
                metadata = {
                    "content": chunk.content,
                    "document_id": chunk.metadata.get("document_id", ""),
                    "chunk_index": str(chunk.metadata.get("chunk_index", 0)),  # Convert to string
                    "document_url": chunk.metadata.get("document_url", ""),
                    "word_count": chunk.metadata.get("word_count", 0)
                }
                
                # Remove sections if it's not a simple type
                if isinstance(chunk.metadata.get("sections"), dict):
                    # Convert sections to string representation
                    metadata["sections"] = str(chunk.metadata.get("sections", {}))
                
                vector = {
                    "id": chunk.chunk_id,
                    "values": embeddings[i],
                    "metadata": metadata
                }
                vectors.append(vector)
            
            # Upsert vectors in batches
            batch_size = 100
            total_batches = (len(vectors) + batch_size - 1) // batch_size
            logger.info(f"📦 Upserting {len(vectors)} vectors in {total_batches} batches...")
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                logger.info(f"📤 Processing batch {batch_num}/{total_batches} ({len(batch)} vectors)...")
                self.index.upsert(vectors=batch)
            
            logger.info(f"✅ Successfully stored {len(chunks)} chunks in Pinecone vector database")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error storing chunks in Pinecone: {e}")
            raise Exception(f"Failed to store chunks in Pinecone: {str(e)}")
    
    def search_similar(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using semantic similarity"""
        try:
            logger.info(f"🔍 Searching Pinecone for query: '{query[:50]}...'")
            
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            logger.info(f"📊 Generated query embedding successfully")
            
            # Search in Pinecone
            top_k = top_k or self.config.TOP_K_RESULTS
            logger.info(f"🔎 Querying Pinecone with top_k={top_k}...")
            
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Process results
            similar_chunks = []
            for match in results.matches:
                if match.score >= self.config.SIMILARITY_THRESHOLD:
                    similar_chunks.append({
                        "content": match.metadata.get("content", ""),
                        "score": match.score,
                        "metadata": match.metadata,
                        "chunk_id": match.id
                    })
            
            logger.info(f"✅ Found {len(similar_chunks)} similar chunks for query using Pinecone (threshold: {self.config.SIMILARITY_THRESHOLD})")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"❌ Error searching similar chunks in Pinecone: {e}")
            raise Exception(f"Failed to search chunks in Pinecone: {str(e)}")
    
    def search_universal(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Universal search strategy for ANY question type"""
        try:
            # For Pinecone, we'll use the same semantic search as it's already optimized
            # This method provides a unified interface for different search strategies
            return self.search_similar(query, top_k)
            
        except Exception as e:
            logger.error(f"❌ Error in universal search in Pinecone: {e}")
            raise Exception(f"Failed to perform universal search in Pinecone: {str(e)}")
    
    def delete_document_chunks(self, document_id: str) -> bool:
        """Delete all chunks for a specific document"""
        try:
            # Query to find chunks for the document
            results = self.index.query(
                vector=[0] * 384,  # Dummy vector
                top_k=1000,
                include_metadata=True,
                filter={"document_id": document_id}
            )
            
            # Delete found chunks
            chunk_ids = [match.id for match in results.matches]
            if chunk_ids:
                self.index.delete(ids=chunk_ids)
                logger.info(f"✅ Deleted {len(chunk_ids)} chunks for document {document_id} from Pinecone")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting document chunks from Pinecone: {e}")
            raise Exception(f"Failed to delete document chunks from Pinecone: {str(e)}")
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "index_name": self.config.PINECONE_INDEX_NAME,
                "dimension": stats.dimension,
                "status": "connected"
            }
        except Exception as e:
            logger.error(f"❌ Error getting index stats from Pinecone: {e}")
            raise Exception(f"Failed to get index stats from Pinecone: {str(e)}")