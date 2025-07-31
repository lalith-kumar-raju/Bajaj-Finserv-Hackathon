# HackRx 6.0 - LLM-Powered Query Retrieval System

An intelligent document processing and query retrieval system for insurance policies, built with FastAPI, Azure OpenAI, and Pinecone.

## 🚀 Features

- **Document Processing**: PDF, DOCX, Email support
- **Semantic Search**: Advanced vector-based document retrieval
- **LLM Integration**: Azure OpenAI GPT-4o for intelligent responses
- **Query Analysis**: Intent classification and entity extraction
- **Caching**: Optimized performance with document caching
- **API**: RESTful API with comprehensive endpoints

## 📁 Project Structure

```
Bajaj-Finserv-Hackathon/
├── main.py                 # FastAPI application
├── config.py              # Configuration settings
├── models.py              # Pydantic data models
├── llm_processor.py       # Azure OpenAI integration
├── vector_store.py        # Pinecone vector database
├── query_analyzer.py      # Query analysis and intent classification
├── document_processor.py  # Document processing and chunking
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker containerization
├── startup.sh            # Azure deployment script
├── .deployment           # Azure deployment config
├── web.config            # Azure web configuration
└── .gitignore           # Git ignore rules
```

## 🛠️ Local Development

### Prerequisites
- Python 3.9+
- Azure OpenAI API key
- Pinecone API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Bajaj-Finserv-Hackathon
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   # Create .env file
   OPENAI_API_KEY=your_azure_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_ENVIRONMENT=your_pinecone_environment
   ```

4. **Run the application:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## 🌐 API Endpoints

### Health Check
```bash
GET /health
```

### System Statistics
```bash
GET /stats
```

### Main Processing
```bash
POST /hackrx/run
Authorization: Bearer your_api_key
Content-Type: application/json

{
  "documents": "https://example.com/policy.pdf",
  "questions": ["What is the grace period?"]
}
```

### File Upload
```bash
POST /hackrx/upload
Authorization: Bearer your_api_key
Content-Type: multipart/form-data

file: your_policy.pdf
questions: ["What is covered?"]
```

## 🚀 Azure Deployment

### Prerequisites
- Azure for Students subscription
- GitHub account
- Azure CLI (optional)

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Ready for Azure deployment"
   git push origin main
   ```

2. **Create Azure App Service:**
   - Go to Azure Portal
   - Create App Service
   - Choose Python 3.9
   - Select Free tier

3. **Configure Environment Variables:**
   - Go to App Service → Configuration
   - Add these variables:
     ```
     OPENAI_API_KEY=your_azure_openai_key
     PINECONE_API_KEY=your_pinecone_key
     PINECONE_ENVIRONMENT=your_pinecone_environment
     HACKRX_API_KEY=031e2883dcfac08106d5a9982528deff7dcd207bd1efbca391476ea56fec65ac
     ```

4. **Connect GitHub:**
   - Go to App Service → Deployment Center
   - Choose GitHub
   - Select your repository
   - Enable automatic deployment

5. **Deploy:**
   - Azure will automatically deploy from GitHub
   - Monitor deployment in the Azure portal

### Azure Configuration Files

- **`startup.sh`**: Azure startup script
- **`.deployment`**: Azure deployment configuration
- **`web.config`**: Azure web server configuration

## 🔧 Configuration

### Document Processing
- `CHUNK_SIZE`: 800 words per chunk
- `CHUNK_OVERLAP`: 150 words overlap
- `MAX_CHUNKS_PER_DOCUMENT`: 500 chunks

### Search Settings
- `TOP_K_RESULTS`: 10 results
- `SIMILARITY_THRESHOLD`: 0.5

### Model Settings
- `LLM_MODEL`: gpt-4o
- `MAX_TOKENS`: 4000
- `TEMPERATURE`: 0.1

## 📊 Performance

- **Document Processing**: Up to 400,000 words
- **Response Time**: 5-15 seconds per query
- **Concurrent Requests**: 10 simultaneous
- **Cache TTL**: 1 hour

## 🔒 Security

- **API Key Authentication**: Bearer token required
- **Input Validation**: Comprehensive request validation
- **Error Handling**: Secure error responses
- **Rate Limiting**: Built-in request limiting

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### API Test
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 031e2883dcfac08106d5a9982528deff7dcd207bd1efbca391476ea56fec65ac" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is the grace period?"]
  }'
```

## 📈 Monitoring

- **Azure Application Insights**: Built-in monitoring
- **Logs**: Available in Azure portal
- **Metrics**: Request count, response time, errors

## 🆘 Troubleshooting

### Common Issues

1. **Port Issues**: Azure uses environment variable `PORT`
2. **Memory Limits**: Free tier has 1GB RAM limit
3. **Timeout**: Requests timeout after 30 seconds
4. **API Keys**: Ensure all environment variables are set

### Debug Commands

```bash
# Check logs
az webapp log tail --name your-app-name --resource-group your-rg

# Restart app
az webapp restart --name your-app-name --resource-group your-rg
```

## 📝 License

This project is part of the HackRx 6.0 competition.

## 🤝 Support

For issues and questions:
1. Check Azure portal logs
2. Review application insights
3. Test locally first
4. Verify environment variables

---

**Deployment Status**: ✅ Ready for Azure deployment
**Cost**: $0/month (Free tier)
**Performance**: Optimized for production use 