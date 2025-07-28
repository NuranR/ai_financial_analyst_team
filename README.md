# A-FIN: Autonomous Financial Information Nexus

A multi-agent system that emulates a team of financial analysts to produce comprehensive investment briefs on publicly traded companies.

## Project Overview

A-FIN consists of four specialized AI agents:

1. **Data Journalist**: Processes news articles, press releases, and social media mentions
2. **Quantitative Analyst**: Analyzes stock price data and detects trading anomalies
3. **Regulator Specialist**: Interprets SEC filings and financial documents
4. **Lead Analyst**: Synthesizes all information into a comprehensive investment brief

## Tech Stack

- **Framework**: Python with CrewAI for multi-agent orchestration
- **LLMs**: OpenAI GPT-4, Anthropic Claude
- **NLP**: NLTK, spaCy, transformers
- **Data Sources**: Alpha Vantage, NewsAPI, SEC EDGAR
- **ML**: PyTorch for anomaly detection models
- **Web Interface**: Streamlit
- **MLOps**: MLflow, Docker

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (see `.env.example`)
4. Run the application: `streamlit run app.py`

## Project Structure

```
├── agents/                 # Individual agent implementations
├── models/                 # ML models for anomaly detection
├── data/                   # Data processing utilities
├── api/                    # External API integrations
├── web/                    # Streamlit web interface
├── config/                 # Configuration files
├── tests/                  # Unit tests
└── notebooks/              # Development notebooks
```

## Development Phases

- [x] Phase 1: Project Setup & Foundation
- [ ] Phase 2: Individual Agents Development
- [ ] Phase 3: Integration & Deployment
