
# Multi-Retrieval and Multi-Modal RAG Chatbot
### SURGE 2025 | Indian Institute of Technology, Kanpur

## Description
This project, developed during SURGE 2025 at IIT Kanpur under the mentorship of **Prof. Salman Ahmad Khan** (*Department of Chemical Engineering*), presents a **Retrieval-Augmented Generation** (RAG)-based chatbot designed for context-aware retrieval and natural language interaction in research-related queries. The system under directory:
```bash
\pipelines_chatbot_RAG
```
 combines multiple retrieval strategies, dynamic visualization, and integration with LLMs to enhance user interaction with scientific literature and experimental data.

The second folder directory:
```bash 
\classifier_parser_scraper
```
 project also includes a synthesis-paragraph scraper and classifier tool that automates the extraction and classification of relevant synthesis-related content from scientific articles.

## Prerequisites
To run or modify this project, ensure the following libraries and APIs are installed/configured:

``` bash
pip install sentence-transformers \
            pinecone-client \
            selenium \
            beautifulsoup4 \
            undetected-chromedriver \
            requests \
            google-generativeai \
            matplotlib \
            numpy==1.24.4 \
            scikit-learn
```

**Also needs:**

google API credentials
Pinecone API credentials
Springer API credentials
modernbert via sentence-transformers

# Results
## RAG Chatbot System
### Retrieval Pipelines:

- **Semantic Search:** Embedding-based nearest-neighbor retrieval
- **Deep Search:** LLM-enhanced keyword + semantic retrieval
- **History-aware Deep Search:** Context retention for multi-turn queries
- **Dynamic Histogram Pipeline:** Enables trend visualization of experimental data in real-time


### Models:
**Embedding finalized for product:** ModernBERT

**LLM finalized for product:** Gemini-1.5 flash (via API)

### Deployment & Validation:
- Web app deployed.
- Retrieval results validated by a domain expert.
- 9,000+ paragraph dataset supported with seamless scaling and low latency

## Synthesis Paragraph Scraper and Classifier
### Web Scraping:

- HTML parser using Selenium + BeautifulSoup

- Integrated with Springer API for domain-specific scraping

### Data Curation:

- 2,000+ paragraph annotations

- minority-class samples added for balance


### Performance:

**F1 Score: 0.96**

**Accuracy: 98%**

**AUC-ROC: 0.9974**

