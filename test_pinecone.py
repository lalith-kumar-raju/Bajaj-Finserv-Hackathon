#!/usr/bin/env python3
"""
Test script to verify Pinecone integration is working correctly.
This script tests the vector store initialization and basic operations.
"""

import os
import sys
import logging
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pinecone_connection():
    """Test Pinecone connection and basic operations"""
    try:
        logger.info("🧪 Testing Pinecone connection...")
        
        # Test configuration
        config = Config()
        logger.info(f"📋 Configuration loaded:")
        logger.info(f"   - Pinecone API Key: {'✅ Set' if config.PINECONE_API_KEY and config.PINECONE_API_KEY != 'your_pinecone_api_key_here' else '❌ Not set'}")
        logger.info(f"   - Pinecone Environment: {config.PINECONE_ENVIRONMENT}")
        logger.info(f"   - Pinecone Index Name: {config.PINECONE_INDEX_NAME}")
        
        # Test vector store initialization
        from vector_store import VectorStore
        logger.info("🔧 Initializing VectorStore...")
        vector_store = VectorStore()
        
        # Test index stats
        logger.info("📊 Getting index stats...")
        stats = vector_store.get_index_stats()
        logger.info(f"✅ Index stats: {stats}")
        
        # Test embedding generation
        logger.info("🧠 Testing embedding generation...")
        test_texts = ["This is a test document", "Another test document"]
        embeddings = vector_store.generate_embeddings(test_texts)
        logger.info(f"✅ Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
        
        # Test search (if there are any vectors in the index)
        if stats.get("total_vectors", 0) > 0:
            logger.info("🔍 Testing search functionality...")
            results = vector_store.search_similar("test query", top_k=5)
            logger.info(f"✅ Search returned {len(results)} results")
        else:
            logger.info("ℹ️  No vectors in index yet, skipping search test")
        
        logger.info("🎉 All Pinecone tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Pinecone test failed: {e}")
        return False

def main():
    """Main test function"""
    logger.info("🚀 Starting Pinecone integration test...")
    
    success = test_pinecone_connection()
    
    if success:
        logger.info("✅ Pinecone integration is working correctly!")
        sys.exit(0)
    else:
        logger.error("❌ Pinecone integration test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 