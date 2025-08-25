from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from functools import lru_cache
import json

from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, validator
from enum import Enum
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from api.news_api import NewsAPI, MockNewsAPI
from agents.base_agent import BaseAgent, AgentResult
from config.prompts import DATA_JOURNALIST_SYSTEM_PROMPT, DATA_JOURNALIST_SUMMARY_PROMPT
from config.settings import settings

# Include cached api service here (ex:- yfinance)
class CompanyLookupService:
    """Stub for company lookup service."""
    def get_company_name(self, ticker: str) -> str:
        return f"Company_{ticker}"

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
    overall_sentiment: Sentiment
    sentiment_score: float = Field(..., ge=-1, le=1)
    sentiment_confidence: float = Field(..., ge=0, le=1)
    key_themes: List[str]
    potential_catalysts: List[Catalyst]
    summary: str

class NewsArticle(BaseModel):
    title: str
    source: str
    published_at: datetime
    description: str = ""
    url: Optional[HttpUrl] = None
    relevance_score: float = Field(0.0, ge=0, le=1)
    
    @validator('published_at', pre=True)
    def parse_published_at(cls, v):
        if isinstance(v, str):
            v = v.replace("Z", "+00:00")
            return datetime.fromisoformat(v)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v

# --- News Service ---
class NewsService:
    def __init__(self, use_mock: bool = False):
        self.news_api = MockNewsAPI() if use_mock else NewsAPI()

    def get_company_news(self, company_name: str, ticker: str, 
                         days_back: int, max_articles: int) -> List[Dict[str, Any]]:
        return self.news_api.get_company_news(company_name, ticker, days_back, max_articles)

class SentimentAnalyzer:
    def __init__(self):
        try:
            from transformers import pipeline
            self._analyzer = pipeline("sentiment-analysis",
                                      model="yiyanghkust/finbert-tone",
                                      tokenizer="yiyanghkust/finbert-tone")
        except ImportError:
            logger.warning("Transformers not installed. Using fallback analyzer.")
            self._analyzer = None

    def analyze_text(self, text: str) -> Tuple[float, float]:
        if not text or self._analyzer is None:
            return 0.0, 0.5
        try:
            result = self._analyzer(text[:512])[0]
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
    def __init__(self):
        try:
            from bertopic import BERTopic
            self._topic_model = BERTopic(verbose=True)
            self._is_initialized = False
        except ImportError:
            logger.warning("BERTopic not installed. Using simple keyword extraction.")
            self._topic_model = None

    def extract_topics(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        if self._topic_model:
            try:
                if not self._is_initialized:
                    topics, _ = self._topic_model.fit_transform(texts)
                    self._is_initialized = True
                else:
                    topics, _ = self._topic_model.transform(texts)
                topic_info = self._topic_model.get_topic_info()
                return topic_info['Name'].head(5).tolist()
            except Exception as e:
                logger.warning(f"Topic modeling failed: {e}")
                return self._fallback_topic_extraction(texts)
        return self._fallback_topic_extraction(texts)

    def _fallback_topic_extraction(self, texts: List[str]) -> List[str]:
        import re
        from collections import Counter
        words = []
        for text in texts:
            words.extend(re.findall(r'\b[A-Z][a-z]+\b', text))
        counts = Counter(words)
        return [w for w, _ in counts.most_common(5)]


class DataJournalistAgent(BaseAgent):
    def __init__(self, news_service: Optional[NewsService] = None,
                 lookup_service: Optional[CompanyLookupService] = None):
        super().__init__("Data Journalist")
        self.news_service = news_service or NewsService()
        self.lookup_service = lookup_service or CompanyLookupService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        logger.info("DataJournalistAgent initialized.")

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10),
           retry=retry_if_exception_type((ConnectionError, TimeoutError)))
    def analyze(self, ticker: str, **kwargs) -> AgentResult:
        try:
            ticker = self._validate_ticker(ticker)
            company_name = kwargs.get("company_name") or self.lookup_service.get_company_name(ticker)
            days_back = kwargs.get("days_back", 3)
            max_articles = kwargs.get("max_articles", 15)

            articles = self._get_cached_news(company_name, ticker, days_back, max_articles)
            valid_articles = self._validate_articles(articles)
            if not valid_articles:
                return self._create_no_data_result(ticker)

            texts = [f"{a.title}. {a.description}" for a in valid_articles]
            structured_analysis = self._perform_structured_analysis(texts, valid_articles)
            
            # Use BaseAgent LLM call
            summary_prompt = DATA_JOURNALIST_SUMMARY_PROMPT.format(
                company_name=company_name,
                ticker=ticker,
                structured_analysis=json.dumps(structured_analysis.dict(), indent=2)
            )
            llm_summary = self._call_llm(summary_prompt, DATA_JOURNALIST_SYSTEM_PROMPT)
            structured_analysis.summary = llm_summary

            confidence = self._calculate_confidence_score(valid_articles, structured_analysis)
            metadata = self._prepare_metadata(valid_articles, structured_analysis)

            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis=llm_summary,
                confidence_score=confidence,
                structured_data=structured_analysis.dict(),
                metadata=metadata
            )
        except Exception as e:
            logger.error(f"Error in news analysis: {e}")
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis="Analysis failed due to technical error.",
                confidence_score=0.0,
                errors=str(e)
            )

    @lru_cache(maxsize=32)
    def _get_cached_news(self, company_name: str, ticker: str, days_back: int, max_articles: int) -> List[Dict[str, Any]]:
        return self.news_service.get_company_news(company_name, ticker, days_back, max_articles)

    def _validate_articles(self, articles: List[Dict[str, Any]]) -> List[NewsArticle]:
        valid = []
        for a in articles:
            try:
                article = NewsArticle(**a)
                if len(article.title) > 10 and article.relevance_score > 0.1:
                    valid.append(article)
            except Exception:
                continue
        return valid

    def _perform_structured_analysis(self, texts: List[str], articles: List[NewsArticle]) -> NewsAnalysisResult:
        scores, confs = [], []
        for t in texts:
            s, c = self.sentiment_analyzer.analyze_text(t)
            scores.append(s)
            confs.append(c)
        avg_score = float(np.mean(scores)) if scores else 0.0
        avg_conf = float(np.mean(confs)) if confs else 0.0
        overall = (Sentiment.BULLISH if avg_score > 0.1 else
                   Sentiment.BEARISH if avg_score < -0.1 else
                   Sentiment.NEUTRAL)
        return NewsAnalysisResult(
            overall_sentiment=overall,
            sentiment_score=avg_score,
            sentiment_confidence=avg_conf,
            key_themes=self.topic_modeler.extract_topics(texts),
            potential_catalysts=self._detect_catalysts(texts),
            summary=""
        )

    def _detect_catalysts(self, texts: List[str]) -> List[Catalyst]:
        catalysts = []
        keywords = {
            FinancialEventType.EARNINGS: ['earnings', 'quarterly results', 'eps', 'revenue'],
            FinancialEventType.PRODUCT_LAUNCH: ['launch', 'new product', 'announce', 'release'],
            FinancialEventType.MERGER_ACQUISITION: ['acquire', 'merge', 'takeover', 'buyout'],
            FinancialEventType.REGULATORY: ['fda', 'approval', 'regulation', 'investigation'],
        }
        for t in texts:
            t_lower = t.lower()
            for et, kws in keywords.items():
                if any(k in t_lower for k in kws):
                    catalysts.append(Catalyst(
                        description=t[:100] + "..." if len(t) > 100 else t,
                        event_type=et,
                        expected_impact="short-term",
                        confidence=0.7
                    ))
                    break
        return catalysts[:3]
    
    # check args of this with abstraction
    def _calculate_confidence_score(self, articles: List[NewsArticle], analysis: NewsAnalysisResult) -> float:
        if not articles: return 0.1
        factors = {
            "data_quantity": min(len(articles)/10, 1.0)*0.3,
            "data_recency": self._calculate_recency_score(articles)*0.25,
            "data_relevance": np.mean([a.relevance_score for a in articles])*0.2,
            "analysis_confidence": analysis.sentiment_confidence*0.25,
        }
        return min(max(sum(factors.values()), 0.1), 0.95)

    def _calculate_recency_score(self, articles: List[NewsArticle]) -> float:
        now = datetime.now(timezone.utc)
        recent_count = 0
        for a in articles:
            pub = a.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            if (now - pub).days <= 1:
                recent_count += 1
        return min(recent_count / len(articles) * 2, 1.0)

    def _prepare_metadata(self, articles: List[NewsArticle], analysis: NewsAnalysisResult) -> Dict[str, Any]:
        sources = list({a.source for a in articles})
        return {
            "articles_analyzed": len(articles),
            "sources_analyzed": sources[:5],
            "avg_relevance_score": np.mean([a.relevance_score for a in articles]),
            "date_range": {
                "oldest": min(a.published_at for a in articles).isoformat(),
                "newest": max(a.published_at for a in articles).isoformat()
            },
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "finbert-topic-1.0"
        }

    def _create_no_data_result(self, ticker: str) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            company_ticker=ticker,
            analysis="No recent relevant news articles found.",
            confidence_score=0.1,
            metadata={"articles_found": 0},
            errors="No relevant news data available"
        )
