from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from functools import lru_cache
import json

from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, validator
from enum import Enum
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import numpy as np

from api.news_api import NewsAPI, MockNewsAPI
from agents.base_agent import BaseAgent, AgentResult
from config.prompts import DATA_JOURNALIST_SYSTEM_PROMPT, DATA_JOURNALIST_SUMMARY_PROMPT
from config.settings import settings
NewsAPI


class Sentiment(str, Enum):
    BEARISH = "bearish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"

class FinancialEventType(str, Enum):
    EARNINGS = "earnings"
    MERGER_ACQUISITION = "merger_acquisition"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    EXECUTIVE_CHANGE = "executive_change"
    GUIDANCE_UPDATE = "guidance_update"
    PARTNERSHIP = "partnership"
    LAWSUIT = "lawsuit"
    OTHER = "other"

class Catalyst(BaseModel):
    description: str
    event_type: FinancialEventType
    expected_impact: str
    confidence: float = Field(..., ge=0, le=1)

class NewsAnalysisResult(BaseModel):
    """Structured analysis of financial news."""
    overall_sentiment: Sentiment
    sentiment_score: float = Field(..., ge=-1, le=1)  
    sentiment_confidence: float = Field(..., ge=0, le=1)
    key_themes: List[str] = Field(..., description="List of key themes/topics")
    potential_catalysts: List[Catalyst]
    summary: str

class NewsArticle(BaseModel):
    """Validated news article model."""
    title: str
    source: str
    published_at: datetime
    description: str = ""
    url: Optional[HttpUrl] = None
    relevance_score: float = Field(0.0, ge=0, le=1)
    
    @validator('published_at', pre=True)
    def parse_published_at(cls, v):
        if isinstance(v, str):
            # Handle various datetime formats
            v = v.replace('Z', '+00:00')
        return v

# --- Service Interfaces ---
class NewsService:

    def __init__(self, use_mock: bool = False):
        self.news_api = MockNewsAPI() if use_mock else NewsAPI()

    """Abstract interface for fetching news."""
    def get_company_news(self, company_name: str, ticker: str, 
                        days_back: int, max_articles: int) -> List[Dict[str, Any]]:
        return self.news_api.get_company_news(
            company_name=company_name,
            ticker=ticker,
            days_back=days_back,
            max_articles=max_articles
        )

# --- Specialized Analysis Services ---
class SentimentAnalyzer:
    """Specialized sentiment analysis for financial text."""
    def __init__(self):
        # In production: load fine-tuned model like FinBERT
        try:
            # Example using transformers library
            from transformers import pipeline
            self._analyzer = pipeline("sentiment-analysis", 
                                    model="yiyanghkust/finbert-tone",
                                    tokenizer="yiyanghkust/finbert-tone")
        except ImportError:
            logger.warning("Transformers not installed. Using fallback analyzer.")
            self._analyzer = None
    
    def analyze_text(self, text: str) -> tuple[float, float]:
        """Return sentiment score (-1 to 1) and confidence."""
        if not text or self._analyzer is None:
            return 0.0, 0.5
            
        try:
            result = self._analyzer(text[:512])[0]  # Truncate to model limits
            label = result['label'].lower()
            confidence = result['score']
            
            if 'positive' in label:
                return confidence, confidence
            elif 'negative' in label:
                return -confidence, confidence
            else:
                return 0.0, confidence
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            return 0.0, 0.5

class TopicModeler:
    """Extracts key themes from news articles."""
    def __init__(self):
        try:
            # Example using BERTopic
            from bertopic import BERTopic
            self._topic_model = BERTopic(verbose=True)
            self._is_initialized = False
        except ImportError:
            logger.warning("BERTopic not installed. Using simple keyword extraction.")
            self._topic_model = None
    
    def extract_topics(self, texts: List[str]) -> List[str]:
        """Extract key topics from a list of texts."""
        if not texts:
            return []
            
        if self._topic_model:
            try:
                if not self._is_initialized:
                    # Fit on initial data
                    topics, _ = self._topic_model.fit_transform(texts)
                    self._is_initialized = True
                else:
                    # Transform new data
                    topics, _ = self._topic_model.transform(texts)
                
                # Get most frequent topics
                topic_info = self._topic_model.get_topic_info()
                return topic_info['Name'].head(5).tolist()
            except Exception as e:
                logger.warning(f"Topic modeling failed: {e}")
                return self._fallback_topic_extraction(texts)
        else:
            return self._fallback_topic_extraction(texts)
    
    def _fallback_topic_extraction(self, texts: List[str]) -> List[str]:
        """Simple keyword-based fallback."""
        # Simple implementation - in production, use RAKE, YAKE, or KeyBERT
        import re
        from collections import Counter
        
        # Extract words and count frequencies
        words = []
        for text in texts:
            words.extend(re.findall(r'\b[A-Z][a-z]+\b', text))  # Capitalized words
        
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(5)]

class DataJournalistAgent(BaseAgent):
    """
    Production-grade agent for financial news analysis.
    Uses specialized models for analysis and LLM only for summarization.
    """
    
    def __init__(self, 
                 news_service: Optional[NewsService] = None,
                 lookup_service: Optional[CompanyLookupService] = None):
        super().__init__("Data Journalist")
        
        # Dependency Injection
        self.news_service = news_service or self._create_default_news_service()
        self.lookup_service = lookup_service or self._create_default_lookup_service()
        
        # Specialized analysis components
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        
        logger.info("DataJournalistAgent initialized with specialized analysis pipeline")
    
    def _create_default_news_service(self) -> NewsService:
        """Create default news service based on configuration."""
        # Implementation would return appropriate NewsService
        # based on available API keys
        from services.news_services import NewsAPIService, AlphaVantageNewsService, MockNewsService
        if settings.alpha_vantage_api_key:
            return AlphaVantageNewsService(settings.alpha_vantage_api_key)
        elif settings.news_api_key:
            return NewsAPIService(settings.news_api_key)
        else:
            logger.warning("Using mock news service - configure API keys for production use")
            return MockNewsService()
    
    def _create_default_lookup_service(self) -> CompanyLookupService:
        """Create default company lookup service."""
        from services.company_services import YahooLookupService, PolygonLookupService
        if settings.polygon_api_key:
            return PolygonLookupService(settings.polygon_api_key)
        else:
            return YahooLookupService()  # Free alternative
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError))
    )
    def analyze(self, ticker: str, **kwargs) -> AgentResult:
        """
        Analyze financial news using specialized models and LLM summarization.
        """
        try:
            # Validate and prepare inputs
            ticker = self._validate_ticker(ticker)
            company_name = kwargs.get('company_name') or self.lookup_service.get_company_name(ticker)
            days_back = kwargs.get('days_back', 3)  # Default to 3 days for recency
            max_articles = kwargs.get('max_articles', 15)
            
            logger.info(f"Starting enhanced news analysis for {ticker}")
            
            # Fetch and validate news articles (with caching)
            articles = self._get_cached_news(company_name, ticker, days_back, max_articles)
            valid_articles = self._validate_articles(articles)
            
            if not valid_articles:
                return self._create_no_data_result(ticker)
            
            # Extract text for analysis
            article_texts = self._prepare_article_texts(valid_articles)
            
            # Perform specialized analysis
            structured_analysis = self._perform_structured_analysis(article_texts, valid_articles)
            
            # Generate LLM summary from structured analysis
            llm_summary = self._generate_llm_summary(structured_analysis, company_name, ticker)
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_confidence_score(valid_articles, structured_analysis)
            
            # Prepare metadata
            metadata = self._prepare_metadata(valid_articles, structured_analysis)
            
            logger.info(f"Completed enhanced analysis for {ticker} with confidence {confidence:.2f}")
            
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis=llm_summary,
                confidence_score=confidence,
                structured_data=structured_analysis.dict(),
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced news analysis for {ticker}: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis="Analysis failed due to technical error.",
                confidence_score=0.0,
                errors=str(e)
            )
    
    @lru_cache(maxsize=32)
    def _get_cached_news(self, company_name: str, ticker: str, 
                        days_back: int, max_articles: int) -> List[Dict[str, Any]]:
        """Cache news results for 15 minutes to avoid redundant API calls."""
        logger.debug(f"Fetching news for {ticker} from service")
        return self.news_service.get_company_news(
            company_name=company_name,
            ticker=ticker,
            days_back=days_back,
            max_articles=max_articles
        )
    
    def _validate_articles(self, articles: List[Dict[str, Any]]) -> List[NewsArticle]:
        """Validate and clean news articles."""
        valid_articles = []
        for article in articles:
            try:
                valid_article = NewsArticle(**article)
                # Filter out very short or irrelevant articles
                if len(valid_article.title) > 10 and valid_article.relevance_score > 0.1:
                    valid_articles.append(valid_article)
            except Exception as e:
                logger.debug(f"Skipping invalid article: {e}")
        return valid_articles
    
    def _prepare_article_texts(self, articles: List[NewsArticle]) -> List[str]:
        """Prepare article texts for analysis."""
        return [f"{article.title}. {article.description}" for article in articles]
    
    def _perform_structured_analysis(self, article_texts: List[str], 
                                   articles: List[NewsArticle]) -> NewsAnalysisResult:
        """Perform structured analysis using specialized models."""
        
        # Analyze sentiment for each article
        sentiment_scores = []
        sentiment_confidences = []
        
        for text in article_texts:
            score, confidence = self.sentiment_analyzer.analyze_text(text)
            sentiment_scores.append(score)
            sentiment_confidences.append(confidence)
        
        # Calculate overall sentiment
        avg_score = np.mean(sentiment_scores) if sentiment_scores else 0
        avg_confidence = np.mean(sentiment_confidences) if sentiment_confidences else 0
        
        if avg_score > 0.1:
            overall_sentiment = Sentiment.BULLISH
        elif avg_score < -0.1:
            overall_sentiment = Sentiment.BEARISH
        else:
            overall_sentiment = Sentiment.NEUTRAL
        
        # Extract key topics
        key_topics = self.topic_modeler.extract_topics(article_texts)
        
        # Detect catalysts (simplified - in production, use NER model)
        catalysts = self._detect_catalysts(article_texts)
        
        return NewsAnalysisResult(
            overall_sentiment=overall_sentiment,
            sentiment_score=float(avg_score),
            sentiment_confidence=float(avg_confidence),
            key_themes=key_topics,
            potential_catalysts=catalysts,
            summary=""  # Will be filled by LLM
        )
    
    def _detect_catalysts(self, texts: List[str]) -> List[Catalyst]:
        """Detect potential market catalysts from text."""
        catalysts = []
        catalyst_keywords = {
            FinancialEventType.EARNINGS: ['earnings', 'quarterly results', 'eps', 'revenue'],
            FinancialEventType.PRODUCT_LAUNCH: ['launch', 'new product', 'announce', 'release'],
            FinancialEventType.MERGER_ACQUISITION: ['acquire', 'merge', 'takeover', 'buyout'],
            FinancialEventType.REGULATORY: ['fda', 'approval', 'regulation', 'investigation'],
        }
        
        for text in texts:
            text_lower = text.lower()
            for event_type, keywords in catalyst_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    catalysts.append(Catalyst(
                        description=text[:100] + "..." if len(text) > 100 else text,
                        event_type=event_type,
                        expected_impact="short-term",
                        confidence=0.7
                    ))
                    break
        
        return catalysts[:3]  # Return top 3 catalysts
    
    def _generate_llm_summary(self, analysis: NewsAnalysisResult, 
                            company_name: str, ticker: str) -> str:
        """Generate narrative summary from structured analysis using LLM."""
        prompt = DATA_JOURNALIST_SUMMARY_PROMPT.format(
            company_name=company_name,
            ticker=ticker,
            structured_analysis=json.dumps(analysis.dict(), indent=2)
        )
        
        return self._call_llm(prompt, DATA_JOURNALIST_SYSTEM_PROMPT)
    
    def _calculate_confidence_score(self, articles: List[NewsArticle], 
                                  analysis: NewsAnalysisResult) -> float:
        """Calculate comprehensive confidence score."""
        if not articles:
            return 0.1
        
        # Weighted factors
        factors = {
            'data_quantity': min(len(articles) / 10, 1.0) * 0.3,
            'data_recency': self._calculate_recency_score(articles) * 0.25,
            'data_relevance': np.mean([a.relevance_score for a in articles]) * 0.2,
            'analysis_confidence': analysis.sentiment_confidence * 0.25,
        }
        
        confidence = sum(factors.values())
        return min(max(confidence, 0.1), 0.95)  # Keep within reasonable bounds
    
    def _calculate_recency_score(self, articles: List[NewsArticle]) -> float:
        """Calculate how recent the articles are."""
        if not articles:
            return 0.0
        
        recent_count = 0
        for article in articles:
            days_old = (datetime.now() - article.published_at.replace(tzinfo=None)).days
            if days_old <= 1:  # Within 1 day
                recent_count += 1
        
        return min(recent_count / len(articles) * 2, 1.0)  # Boost recent articles
    
    def _prepare_metadata(self, articles: List[NewsArticle], 
                         analysis: NewsAnalysisResult) -> Dict[str, Any]:
        """Prepare comprehensive metadata."""
        sources = list(set(article.source for article in articles))
        
        return {
            'articles_analyzed': len(articles),
            'sources_analyzed': sources[:5],
            'avg_relevance_score': np.mean([a.relevance_score for a in articles]),
            'date_range': {
                'oldest': min(a.published_at for a in articles).isoformat(),
                'newest': max(a.published_at for a in articles).isoformat()
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'model_version': 'finbert-topic-1.0'
        }
    
    def _create_no_data_result(self, ticker: str) -> AgentResult:
        """Create result for when no news data is available."""
        return AgentResult(
            agent_name=self.name,
            company_ticker=ticker,
            analysis="No recent relevant news articles found for analysis.",
            confidence_score=0.1,
            metadata={'articles_found': 0},
            errors="No relevant news data available"
        )