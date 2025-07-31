from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import time
import logging
import requests
import hashlib
import re
import io
import os
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# Import real components
from config import Config
from models import HackRxRequest, HackRxResponse, QueryAnalysis, QueryIntent

# Import document processing components
import PyPDF2
import fitz  # PyMuPDF
from docx import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()

# Initialize FastAPI app
app = FastAPI(
    title="HackRx 6.0 - LLM-Powered Query Retrieval System",
    description="Intelligent document processing and query retrieval system for insurance policies",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "HackRx",
            "description": "Main HackRx API endpoints for document processing and query retrieval"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import configuration from config.py
from config import Config

# Enhanced Document Processor
class EnhancedDocumentProcessor:
    def __init__(self):
        self.config = Config()
    
    def download_document(self, url: str) -> str:
        """Download document and extract text from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            content = response.content
            
            # Determine file type and extract text
            if url.lower().endswith('.pdf'):
                return self._extract_text_from_pdf(content)
            elif url.lower().endswith(('.docx', '.doc')):
                return self._extract_text_from_docx(content)
            elif 'email' in url.lower() or url.lower().endswith('.eml'):
                return self._extract_text_from_email(content)
            else:
                # Try PDF as default
                return self._extract_text_from_pdf(content)
                
        except Exception as e:
            logger.error(f"Error downloading document from {url}: {e}")
            raise
    
    def process_uploaded_file(self, file: UploadFile) -> str:
        """Process uploaded file and extract text"""
        try:
            content = file.file.read()
            file_extension = file.filename.lower().split('.')[-1]
            
            if file_extension == 'pdf':
                return self._extract_text_from_pdf(content)
            elif file_extension in ['docx', 'doc']:
                return self._extract_text_from_docx(content)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error processing uploaded file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    def _extract_text_from_pdf(self, content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            # Try PyMuPDF first (better text extraction)
            doc = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return self._clean_text(text)
        except Exception as e:
            logger.warning(f"PyMuPDF failed, trying PyPDF2: {e}")
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return self._clean_text(text)
            except Exception as e2:
                logger.error(f"Both PDF extractors failed: {e2}")
                raise HTTPException(status_code=500, detail="Failed to extract text from PDF")
    
    def _extract_text_from_docx(self, content: bytes) -> str:
        """Extract text from DOCX content"""
        try:
            doc = Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from DOCX")
    
    def _extract_text_from_email(self, content: bytes) -> str:
        """Extract text from email content"""
        try:
            import email
            msg = email.message_from_bytes(content)
            text = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode()
            else:
                text = msg.get_payload(decode=True).decode()
            
            return self._clean_text(text)
        except Exception as e:
            logger.error(f"Error extracting text from email: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract text from email")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\{\}]', '', text)
        # Normalize line breaks
        text = text.replace('\n', ' ').replace('\r', ' ')
        return text.strip()
    
    def chunk_text(self, text: str, document_id: str) -> List[Dict[str, Any]]:
        """Enhanced semantic chunking with better boundaries"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split(' ')
        
        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                continue
            
            # Clean paragraph
            paragraph = self._clean_text(paragraph)
            if len(paragraph.strip()) < 30:
                continue
            
            # Split long paragraphs into smaller chunks
            if len(paragraph) > self.config.CHUNK_SIZE:
                sub_chunks = self._split_long_paragraph(paragraph)
                for sub_idx, sub_chunk in enumerate(sub_chunks):
                    chunk_id = f"{document_id}_para_{para_idx}_sub_{sub_idx}"
                    chunks.append({
                        "content": sub_chunk,
                        "chunk_id": chunk_id,
                        "metadata": {
                            "document_id": document_id,
                            "paragraph_index": para_idx,
                            "sub_chunk_index": sub_idx,
                            "chunk_type": "paragraph_sub",
                            "word_count": len(sub_chunk.split())
                        }
                    })
            else:
                chunk_id = f"{document_id}_para_{para_idx}"
                chunks.append({
                    "content": paragraph,
                    "chunk_id": chunk_id,
                    "metadata": {
                        "document_id": document_id,
                        "paragraph_index": para_idx,
                        "chunk_type": "paragraph",
                        "word_count": len(paragraph.split())
                    }
                })
        
        # Limit chunks to prevent memory issues
        return chunks[:self.config.MAX_CHUNKS_PER_DOCUMENT]
    
    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """Split long paragraphs into smaller chunks with overlap"""
        words = paragraph.split()
        chunks = []
        
        chunk_size = self.config.CHUNK_SIZE
        overlap = self.config.CHUNK_OVERLAP
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) >= 50:
                chunks.append(chunk_text)
        
        return chunks

# Enhanced Vector Store
class EnhancedVectorStore:
    def __init__(self):
        self.vectors = {}
        self.documents = {}
        self.vector_dimension = 384
        self.cache = {}  # Simple cache for embeddings
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate enhanced embeddings with caching"""
        embeddings = []
        for text in texts:
            # Check cache first
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.cache:
                embeddings.append(self.cache[text_hash])
                continue
            
            # Create enhanced hash-based embedding
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            
            embedding = []
            for i in range(self.vector_dimension):
                byte_idx = i % len(hash_bytes)
                embedding.append(float(hash_bytes[byte_idx]) / 255.0)
            
            # Cache the embedding
            self.cache[text_hash] = embedding
            embeddings.append(embedding)
        
        return embeddings
    
    def search_similar(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Enhanced search with better ranking"""
        try:
            query_embedding = self.generate_embeddings([query])[0]
            
            similarities = []
            for chunk_id, vector in self.vectors.items():
                similarity = self._cosine_similarity(query_embedding, vector)
                similarities.append((chunk_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            top_k = top_k or self.config.TOP_K_RESULTS
            results = []
            
            # Lower threshold for more results
            threshold = self.config.SIMILARITY_THRESHOLD
            
            for chunk_id, score in similarities[:top_k]:
                if score >= threshold:
                    doc = self.documents.get(chunk_id, {})
                    results.append({
                        "content": doc.get("content", ""),
                        "score": score,
                        "metadata": doc.get("metadata", {}),
                        "chunk_id": chunk_id
                    })
            
            # If no results with current threshold, try with lower threshold
            if not results and threshold > 0.3:
                for chunk_id, score in similarities[:top_k]:
                    if score >= 0.3:
                        doc = self.documents.get(chunk_id, {})
                        results.append({
                            "content": doc.get("content", ""),
                            "score": score,
                            "metadata": doc.get("metadata", {}),
                            "chunk_id": chunk_id
                        })
            
            return results
        except Exception as e:
            logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity"""
        try:
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a * a for a in vec1) ** 0.5
            norm2 = sum(a * a for a in vec2) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "total_vectors": len(self.vectors),
            "index_name": "enhanced_vector_store",
            "dimension": self.vector_dimension
        }

    def search_universal(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Universal search strategy for ANY question type"""
        try:
            # Multi-pass search strategy
            results = []
            
            # Pass 1: Semantic search
            semantic_results = self._semantic_search(query, top_k or self.config.TOP_K_RESULTS)
            results.extend(semantic_results)
            
            # Pass 2: Keyword search for better coverage
            keyword_results = self._keyword_search(query, top_k or self.config.TOP_K_RESULTS)
            results.extend(keyword_results)
            
            # Pass 3: Fuzzy search for partial matches
            fuzzy_results = self._fuzzy_search(query, top_k or self.config.TOP_K_RESULTS)
            results.extend(fuzzy_results)
            
            # Remove duplicates and rank by relevance
            unique_results = self._deduplicate_and_rank(results)
            
            return unique_results[:top_k or self.config.TOP_K_RESULTS]
            
        except Exception as e:
            logger.error(f"Error in universal search: {e}")
            return []
    
    def _semantic_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Semantic similarity search"""
        try:
            query_embedding = self.generate_embeddings([query])[0]
            
            similarities = []
            for chunk_id, vector in self.vectors.items():
                similarity = self._cosine_similarity(query_embedding, vector)
                similarities.append((chunk_id, similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for chunk_id, score in similarities[:top_k]:
                if score >= self.config.SIMILARITY_THRESHOLD:
                    doc = self.documents.get(chunk_id, {})
                    results.append({
                        "content": doc.get("content", ""),
                        "score": score,
                        "metadata": doc.get("metadata", {}),
                        "chunk_id": chunk_id,
                        "search_type": "semantic"
                    })
            
            return results
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Keyword-based search for better coverage"""
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            
            results = []
            for chunk_id, doc in self.documents.items():
                content = doc.get("content", "").lower()
                
                # Calculate keyword match score
                keyword_score = 0
                for keyword in keywords:
                    if keyword in content:
                        keyword_score += 1
                
                if keyword_score > 0:
                    # Normalize score
                    normalized_score = keyword_score / len(keywords)
                    
                    results.append({
                        "content": doc.get("content", ""),
                        "score": normalized_score,
                        "metadata": doc.get("metadata", {}),
                        "chunk_id": chunk_id,
                        "search_type": "keyword"
                    })
            
            # Sort by score and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _fuzzy_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fuzzy search for partial matches"""
        try:
            # Extract words from query
            query_words = query.lower().split()
            
            results = []
            for chunk_id, doc in self.documents.items():
                content = doc.get("content", "").lower()
                content_words = content.split()
                
                # Calculate fuzzy match score
                fuzzy_score = 0
                for query_word in query_words:
                    if len(query_word) > 2:  # Only consider words longer than 2 chars
                        for content_word in content_words:
                            if len(content_word) > 2:
                                # Simple fuzzy matching
                                if query_word in content_word or content_word in query_word:
                                    fuzzy_score += 1
                                elif self._similar_words(query_word, content_word):
                                    fuzzy_score += 0.5
                
                if fuzzy_score > 0:
                    # Normalize score
                    normalized_score = fuzzy_score / len(query_words)
                    
                    results.append({
                        "content": doc.get("content", ""),
                        "score": normalized_score,
                        "metadata": doc.get("metadata", {}),
                        "chunk_id": chunk_id,
                        "search_type": "fuzzy"
                    })
            
            # Sort by score and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "what", "how", "when", "where", "why", "which", "who"}
        
        words = query.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _similar_words(self, word1: str, word2: str) -> bool:
        """Check if two words are similar"""
        if len(word1) < 3 or len(word2) < 3:
            return False
        
        # Simple similarity check
        common_chars = sum(1 for c in word1 if c in word2)
        return common_chars >= min(len(word1), len(word2)) * 0.7
    
    def _deduplicate_and_rank(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and rank by relevance"""
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            chunk_id = result["chunk_id"]
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        # Sort by score
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        
        return unique_results

# Enhanced Query Analyzer
class EnhancedQueryAnalyzer:
    def __init__(self):
        self.config = Config()
        # Load spaCy model for NER (fallback to English if not available)
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not available, using basic NER")
            self.nlp = None
        
        # Universal insurance keywords for ANY question type
        self.intent_keywords = {
            QueryIntent.COVERAGE_CHECK: [
                "cover", "coverage", "covered", "include", "policy covers", "eligible", "pay", "reimburse",
                "surgery", "treatment", "procedure", "medical", "hospitalization", "benefit", "provide",
                "what is covered", "what does it cover", "what's included", "what's covered"
            ],
            QueryIntent.WAITING_PERIOD: [
                "waiting period", "wait", "time", "duration", "months", "years", "days",
                "pre-existing", "existing condition", "initial period", "when", "how long",
                "timeframe", "period", "delay", "before coverage"
            ],
            QueryIntent.EXCLUSION_CHECK: [
                "exclude", "exclusion", "not covered", "not include", "restriction", "limitation",
                "not applicable", "what's not covered", "what's excluded", "what's not included",
                "limitations", "restrictions", "not eligible", "not covered under"
            ],
            QueryIntent.CLAIM_PROCESS: [
                "claim", "claim process", "how to claim", "claim procedure", "documentation",
                "submit claim", "claim form", "how to file", "claiming", "reimbursement",
                "claim submission", "claim documentation", "claim requirements"
            ],
            QueryIntent.POLICY_DETAILS: [
                "policy", "details", "information", "what is", "define", "explain", "describe",
                "tell me about", "policy terms", "policy conditions", "policy features",
                "policy benefits", "policy limits", "policy coverage"
            ]
        }
        
        # Universal insurance entities for ANY question
        self.insurance_entities = {
            "age": r"(\d+)\s*(?:year|yr)s?\s*old",
            "procedure": r"(surgery|treatment|procedure|operation)\s*(?:for|of)?\s*([^,\.]+)",
            "location": r"(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            "policy_duration": r"(\d+)\s*(?:month|year)s?\s*(?:old\s+)?policy",
            "amount": r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹|percent|%)",
            "disease": r"(?:for|of|with)\s+([^,\.]+(?:\s+disease|\s+condition))",
            "hospital": r"(?:at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:hospital|clinic|medical center)",
            "insurance_type": r"(?:health|medical|life|accident|travel)\s+insurance",
            "coverage_type": r"(?:inpatient|outpatient|day care|pre|post)\s+(?:treatment|surgery|care)",
            "time_period": r"(\d+)\s*(?:days?|months?|years?)",
            "percentage": r"(\d+)\s*(?:percent|%)",
            "currency": r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)",
            "medical_term": r"(?:surgery|treatment|procedure|operation|therapy|medication|diagnosis)",
            "condition": r"(?:pre-existing|existing|chronic|acute|temporary|permanent)\s+(?:condition|disease|illness)"
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Universal query analysis for ANY type of question"""
        query_lower = query.lower().strip()
        
        # Universal intent classification
        intent = self._classify_universal_intent(query_lower)
        
        # Universal entity extraction
        entities = self._extract_universal_entities(query)
        
        # Universal confidence calculation
        confidence = self._calculate_universal_confidence(query_lower, intent)
        
        # Universal query processing
        processed_query = self._process_universal_query(query, intent, entities)
        
        return QueryAnalysis(
            intent=intent,
            entities=entities,
            confidence=confidence,
            processed_query=processed_query
        )
    
    def _classify_universal_intent(self, query: str) -> QueryIntent:
        """Universal intent classification for ANY question type"""
        intent_scores = {}
        
        # Enhanced scoring for better accuracy
        for intent, keywords in self.intent_keywords.items():
            score = 0
            for keyword in keywords:
                # Exact match gets higher score
                if keyword in query:
                    score += 2
                # Partial match gets lower score
                elif any(word in query for word in keyword.split()):
                    score += 1
            intent_scores[intent] = score
        
        # Find intent with highest score
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            if intent_scores[best_intent] > 0:
                return best_intent
        
        # Universal fallback - analyze question type
        return self._analyze_question_type(query)
    
    def _analyze_question_type(self, query: str) -> QueryIntent:
        """Analyze question type for universal classification"""
        query_lower = query.lower()
        
        # Question word analysis
        question_words = ["what", "how", "when", "where", "why", "which", "who"]
        question_word = next((word for word in question_words if word in query_lower), None)
        
        # Action word analysis
        action_words = {
            "cover": QueryIntent.COVERAGE_CHECK,
            "wait": QueryIntent.WAITING_PERIOD,
            "exclude": QueryIntent.EXCLUSION_CHECK,
            "claim": QueryIntent.CLAIM_PROCESS,
            "define": QueryIntent.POLICY_DETAILS,
            "explain": QueryIntent.POLICY_DETAILS,
            "describe": QueryIntent.POLICY_DETAILS
        }
        
        for action, intent in action_words.items():
            if action in query_lower:
                return intent
        
        # Default to general query for unknown types
        return QueryIntent.GENERAL_QUERY
    
    def _extract_universal_entities(self, query: str) -> Dict[str, Any]:
        """Universal entity extraction for ANY question"""
        entities = {}
        
        # Extract using enhanced regex patterns
        for entity_type, pattern in self.insurance_entities.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                entities[entity_type] = matches
        
        # Use spaCy for additional NER if available
        if self.nlp:
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "CARDINAL", "MONEY", "QUANTITY"]:
                    entities[f"ner_{ent.label_.lower()}"] = ent.text
        
        # Universal entity extraction
        entities.update(self._extract_universal_patterns(query))
        
        return entities
    
    def _extract_universal_patterns(self, query: str) -> Dict[str, Any]:
        """Extract universal patterns from ANY question"""
        patterns = {}
        
        # Extract numbers
        numbers = re.findall(r'\d+', query)
        if numbers:
            patterns["numbers"] = numbers
        
        # Extract amounts
        amounts = re.findall(r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:rs|rupees|inr|₹)', query, re.IGNORECASE)
        if amounts:
            patterns["amounts"] = amounts
        
        # Extract time periods
        time_periods = re.findall(r'(\d+)\s*(?:days?|months?|years?)', query, re.IGNORECASE)
        if time_periods:
            patterns["time_periods"] = time_periods
        
        # Extract medical terms
        medical_terms = re.findall(r'(surgery|treatment|procedure|operation|therapy|medication|diagnosis)', query, re.IGNORECASE)
        if medical_terms:
            patterns["medical_terms"] = medical_terms
        
        # Extract conditions
        conditions = re.findall(r'(pre-existing|existing|chronic|acute|temporary|permanent)', query, re.IGNORECASE)
        if conditions:
            patterns["conditions"] = conditions
        
        return patterns
    
    def _calculate_universal_confidence(self, query: str, intent: QueryIntent) -> float:
        """Universal confidence calculation for ANY question"""
        if intent == QueryIntent.GENERAL_QUERY:
            return 0.7  # Higher base confidence for general queries
        
        keywords = self.intent_keywords.get(intent, [])
        if not keywords:
            return 0.7
        
        matches = sum(1 for keyword in keywords if keyword in query)
        confidence = min(matches / len(keywords), 1.0)
        
        # Boost confidence for specific patterns
        if intent == QueryIntent.COVERAGE_CHECK and any(word in query for word in ["cover", "coverage", "covered"]):
            confidence = min(confidence + 0.2, 1.0)
        elif intent == QueryIntent.WAITING_PERIOD and "waiting period" in query:
            confidence = min(confidence + 0.3, 1.0)
        elif intent == QueryIntent.EXCLUSION_CHECK and "exclusion" in query:
            confidence = min(confidence + 0.2, 1.0)
        
        return confidence
    
    def _process_universal_query(self, query: str, intent: QueryIntent, entities: Dict[str, Any]) -> str:
        """Universal query processing for ANY question type"""
        processed = query
        
        # Add universal context for better retrieval
        if intent == QueryIntent.COVERAGE_CHECK:
            if "cover" not in processed.lower():
                processed += " coverage policy benefits"
        elif intent == QueryIntent.WAITING_PERIOD:
            if "waiting period" not in processed.lower():
                processed += " waiting period time duration"
        elif intent == QueryIntent.EXCLUSION_CHECK:
            if "exclusion" not in processed.lower():
                processed += " exclusion limitation restriction"
        elif intent == QueryIntent.CLAIM_PROCESS:
            if "claim" not in processed.lower():
                processed += " claim process procedure"
        elif intent == QueryIntent.POLICY_DETAILS:
            if "policy" not in processed.lower():
                processed += " policy terms conditions"
        
        # Add universal entity information
        if entities.get("medical_terms"):
            processed += f" {' '.join(entities['medical_terms'])}"
        if entities.get("time_periods"):
            processed += f" {' '.join(entities['time_periods'])}"
        if entities.get("amounts"):
            processed += f" {' '.join(entities['amounts'])}"
        
        return processed.strip()
    
    def get_universal_suggestions(self, query: str) -> List[str]:
        """Generate universal query suggestions for ANY question"""
        suggestions = []
        
        # Add variations with different terms
        base_query = query.lower()
        
        # Universal synonym expansion
        synonyms = {
            "cover": ["coverage", "include", "provide", "offer"],
            "waiting period": ["wait", "time", "duration", "period"],
            "exclude": ["exclusion", "not covered", "restriction", "limitation"],
            "claim": ["claim process", "claim procedure", "claim submission"],
            "policy": ["insurance", "coverage", "plan", "terms"],
            "surgery": ["operation", "procedure", "treatment"],
            "hospital": ["medical center", "clinic", "healthcare facility"],
            "benefit": ["coverage", "advantage", "feature", "provision"]
        }
        
        for original, syns in synonyms.items():
            if original in base_query:
                for syn in syns:
                    suggestions.append(query.replace(original, syn))
        
        # Add question variations
        question_variations = [
            query.replace("what is", "how does"),
            query.replace("how", "what"),
            query.replace("when", "what time"),
            query.replace("where", "in what location")
        ]
        suggestions.extend(question_variations)
        
        return list(set(suggestions))[:8]  # Limit to 8 unique suggestions

# Enhanced LLM Processor (with real OpenAI integration when API key is available)
class EnhancedLLMProcessor:
    def __init__(self):
        self.config = Config()
        self.use_openai = False
        
        # Try to initialize OpenAI if API key is available
        try:
            import openai
            if self.config.OPENAI_API_KEY and self.config.OPENAI_API_KEY != "your_openai_api_key_here":
                openai.api_key = self.config.OPENAI_API_KEY
                self.use_openai = True
                logger.info("OpenAI API initialized successfully")
            else:
                logger.info("OpenAI API key not configured, using enhanced mock responses")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI: {e}")
    
    def process_query(self, query: str, relevant_chunks: List[Dict[str, Any]], query_analysis: QueryAnalysis) -> str:
        """Universal query processing for ANY question type"""
        try:
            if not self.use_openai:
                raise Exception("OpenAI API key not configured. Please set your OpenAI API key in the .env file.")
            
            # Use universal OpenAI processing
            return self._process_universal_with_openai(query, relevant_chunks, query_analysis)
                
        except Exception as e:
            logger.error(f"Error processing query with LLM: {e}")
            return f"I apologize, but I encountered an error while processing your query: {str(e)}. Please ensure your OpenAI API key is properly configured."
    
    def _process_universal_with_openai(self, query: str, relevant_chunks: List[Dict[str, Any]], query_analysis: QueryAnalysis) -> str:
        """Universal processing with adaptive prompts for ANY question"""
        from openai import AzureOpenAI
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            api_key=self.config.OPENAI_API_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint="https://bajaj-model-gpt-4.openai.azure.com/"
        )
        
        # Enhanced context preparation
        context = self._prepare_universal_context(relevant_chunks)
        
        # Universal system prompt for ANY question type
        universal_system_prompt = """You are an expert insurance policy analyzer. Your task is to answer ANY type of question about insurance policies based on the provided document content.

Key Guidelines:
1. Answer ANY type of question - coverage, waiting periods, exclusions, claims, policy details, definitions, procedures, amounts, timeframes, conditions, etc.
2. Base answers ONLY on the provided policy document content
3. Be specific and cite exact clauses when possible
4. If information is not in the document, say so clearly
5. Provide clear, accurate answers for ANY question format
6. Use professional insurance terminology
7. If partial information is available, state what you know and what's unclear
8. Handle questions about amounts, percentages, timeframes, procedures, conditions, etc.
9. Answer questions about definitions, explanations, descriptions, comparisons, etc.
10. Provide comprehensive answers that directly address the question

CRITICAL FORMATTING RULES - FOLLOW EXACTLY:
- Write in PLAIN TEXT ONLY - like a simple text message
- NO markdown formatting AT ALL
- NO bold, NO italic, NO special characters
- NO escape characters like \\n or \\"
- NO line breaks or paragraph breaks
- NO bullet points or lists
- NO quotes with escape characters
- Write everything as ONE continuous paragraph
- Use simple periods and commas only
- Write as if you're sending a basic SMS text

ANSWER LENGTH RULES:
- Keep answers CONCISE and DIRECT
- Answer the question in 1-2 sentences maximum
- NO verbose explanations
- NO "information not found" messages
- If you can't answer, just say "Not specified in the policy"
- Focus on the specific information requested

IMPORTANT: Your response must be completely plain text with no formatting whatsoever."""

        # Enhanced user prompt with universal context
        user_prompt = self._create_universal_user_prompt(query, context, query_analysis)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": universal_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,  # Increased for comprehensive answers
            temperature=0.1
        )
        
        # Get the response and clean it
        response_text = response.choices[0].message.content.strip()
        
        # Simple cleaning - just remove any remaining formatting
        cleaned_response = self._simple_clean(response_text)
        
        return cleaned_response
    
    def _simple_clean(self, text: str) -> str:
        """Simple cleaning to remove any formatting"""
        import re
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Remove code blocks
        
        # Remove ALL newlines and replace with spaces
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        # Remove any remaining escape characters
        text = text.replace('\\n', ' ')
        text = text.replace('\\t', ' ')
        text = text.replace('\\"', '"')
        
        return text.strip()
    
    def _prepare_universal_context(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Universal context preparation for ANY question type"""
        if not relevant_chunks:
            return "No relevant policy information found in the document."
        
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            content = chunk.get("content", "")
            score = chunk.get("score", 0)
            search_type = chunk.get("search_type", "unknown")
            
            # Include all relevant chunks regardless of search type
            if score >= 0.3:  # Lower threshold for universal coverage
                context_parts.append(f"Policy Section {i} (Relevance: {score:.2f}, Search: {search_type}): {content}")
        
        if not context_parts:
            return "Limited relevant policy information found. Please refer to the complete policy document for accurate details."
        
        return " ".join(context_parts)
    
    def _create_universal_user_prompt(self, query: str, context: str, query_analysis: QueryAnalysis) -> str:
        """Universal user prompt for ANY question type"""
        entities_info = ""
        if query_analysis.entities:
            entities_info = f" Extracted Information: {query_analysis.entities}"
        
        prompt = f"""Based on the following insurance policy document sections, please answer this question:

Question: {query}

{entities_info}

Policy Document Sections:
{context}

Instructions:
1. **Answer ANY type of question** - coverage, waiting periods, exclusions, claims, policy details, definitions, procedures, amounts, timeframes, conditions, etc.
2. **Base your answer ONLY on the provided policy document sections**
3. **Be specific and accurate**
4. **If the information is not in the provided sections, clearly state this**
5. **If partial information is available, state what you know and what's unclear**
6. **Use professional insurance terminology**
7. **Provide comprehensive information that directly addresses the question**
8. **Handle questions about numbers, percentages, timeframes, procedures, etc.**
9. **Answer questions about definitions, explanations, descriptions, etc.**
10. **Provide clear, helpful information regardless of question format**

Please provide a comprehensive answer that directly addresses the question."""

        return prompt


# Initialize components
config = Config()
document_processor = EnhancedDocumentProcessor()

# Use real Pinecone instead of fallback
try:
    from vector_store import VectorStore
    vector_store = VectorStore()
    logger.info("Using real Pinecone vector store")
except Exception as e:
    logger.warning(f"Falling back to in-memory vector store: {e}")
    vector_store = EnhancedVectorStore()

query_analyzer = EnhancedQueryAnalyzer()
llm_processor = EnhancedLLMProcessor()

# Add caching for processed documents
document_cache = {}
embedding_cache = {}

# Cache for processed documents
def get_cached_document(document_id: str) -> Optional[Dict[str, Any]]:
    """Get cached document if available"""
    return document_cache.get(document_id)

def cache_document(document_id: str, chunks: List[Dict[str, Any]]):
    """Cache processed document"""
    document_cache[document_id] = {
        "chunks": chunks,
        "timestamp": time.time()
    }

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")
    
    api_key = authorization.replace("Bearer ", "")
    if api_key != config.HACKRX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return api_key

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "HackRx 6.0 - LLM-Powered Query Retrieval System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "stats": "/stats"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = vector_store.get_index_stats()
        
        return {
            "status": "healthy",
            "vector_store": "connected" if stats.get("total_vectors", 0) >= 0 else "disconnected",
            "llm_processor": "ready",
            "document_processor": "ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get("/stats")
async def get_stats():
    """Get enhanced system statistics"""
    try:
        vector_stats = vector_store.get_index_stats()
        
        return {
            "vector_database": vector_stats,
            "cache_size": len(document_cache),
            "embedding_cache_size": len(embedding_cache),
            "config": {
                "embedding_model": "all-MiniLM-L6-v2",
                "llm_model": "gpt-4o" if llm_processor.use_openai else "enhanced_mock",
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k_results": config.TOP_K_RESULTS,
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
                "max_chunks_per_document": config.MAX_CHUNKS_PER_DOCUMENT
            },
            "performance": {
                "cached_documents": len(document_cache),
                "total_embeddings": vector_stats.get("total_vectors", 0)
            }
        }
    except Exception as e:
        logger.error(f"Stats endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.post("/hackrx/run", tags=["HackRx"])
async def run_hackrx(
    request: HackRxRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Main HackRx endpoint for processing queries"""
    start_time = time.time()
    
    try:
        # Verify API key
        if credentials.credentials != config.HACKRX_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Check if document is already processed
        document_id = hashlib.md5(request.documents.encode()).hexdigest()
        
        if document_id not in document_cache:
            logger.info(f"Processing new document: {request.documents}")
            
            # Download and process document
            document_text = document_processor.download_document(request.documents)
            chunks = document_processor.chunk_text(document_text, document_id)
            
            # Store chunks in vector database
            vector_store.store_chunks(chunks)
            
            # Cache the processed document
            document_cache[document_id] = {
                "url": request.documents,
                "chunks_count": len(chunks),
                "processed_at": time.time()
            }
        else:
            logger.info(f"Using cached document: {request.documents}")
        
        # Process each question
        answers = []
        for question in request.questions:
            # Analyze query
            query_analysis = query_analyzer.analyze_query(question)
            
            # Search for relevant chunks
            relevant_chunks = vector_store.search_similar(question, config.TOP_K_RESULTS)
            
            # Process with LLM
            answer = llm_processor.process_query(question, relevant_chunks, query_analysis)
            answers.append(answer)
        
        processing_time = time.time() - start_time
        
        return {
            "answers": answers
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/api/v1/hackrx/run", tags=["HackRx"])
async def run_hackrx_v1(
    request: HackRxRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Alternative endpoint for HackRx processing"""
    return await run_hackrx(request, credentials)

@app.post("/hackrx/upload", tags=["HackRx"])
async def upload_and_process(
    file: UploadFile = File(...),
    questions: str = Form(...),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Universal document processing for ANY question type"""
    start_time = time.time()
    
    try:
        # Verify API key
        if credentials.credentials != config.HACKRX_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_extension = file.filename.lower().split('.')[-1]
        if file_extension not in ['pdf', 'docx', 'doc']:
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        # Generate document ID from filename
        document_id = hashlib.md5(file.filename.encode()).hexdigest()
        
        # Check cache first
        cached_doc = get_cached_document(document_id)
        if cached_doc:
            logger.info(f"Using cached document: {file.filename}")
            chunks = cached_doc["chunks"]
        else:
            # Process uploaded file
            logger.info(f"Processing uploaded file: {file.filename}")
            document_text = document_processor.process_uploaded_file(file)
            
            # Process document chunks
            chunks = document_processor.chunk_text(document_text, document_id)
            
            # Cache the processed document
            cache_document(document_id, chunks)
        
        # Convert dictionary chunks to DocumentChunk objects for vector store
        from models import DocumentChunk
        document_chunks = []
        for chunk in chunks:
            doc_chunk = DocumentChunk(
                content=chunk["content"],
                metadata=chunk["metadata"],
                chunk_id=chunk["chunk_id"]
            )
            document_chunks.append(doc_chunk)
        
        # Store chunks in vector database
        vector_store.store_chunks(document_chunks)
        
        # Parse questions
        try:
            import json
            if questions.strip().startswith('[') and questions.strip().endswith(']'):
                questions_list = json.loads(questions)
            else:
                questions_list = [q.strip().strip('"\'') for q in questions.split(',')]
            
            if not isinstance(questions_list, list):
                raise ValueError("Questions must be a list")
            if not questions_list:
                raise ValueError("At least one question is required")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid questions format. Error: {str(e)}")
        
        # Process each question with universal approach
        answers = []
        for question in questions_list:
            # Universal query analysis
            query_analysis = query_analyzer.analyze_query(question)
            
            # Universal search for relevant chunks
            relevant_chunks = vector_store.search_universal(question, config.TOP_K_RESULTS)
            
            # Universal LLM processing
            answer = llm_processor.process_query(question, relevant_chunks, query_analysis)
            answers.append(answer)
        
        processing_time = time.time() - start_time
        
        return {
            "answers": answers,
            "processing_time": processing_time,
            "chunks_processed": len(chunks),
            "cache_used": cached_doc is not None,
            "universal_processing": True
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import os
    
    # Get port from environment variable (for Render) or default to 8000
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(app, host="0.0.0.0", port=port) 