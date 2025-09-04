from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
from functools import lru_cache
import json
import re
from collections import Counter

from loguru import logger
from pydantic import BaseModel, Field, HttpUrl, field_validator
from enum import Enum
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from api.news_api import NewsAPI, MockNewsAPI
from api.company_loookup_api import CompanyLookupAPI, MockCompanyLookupAPI
from agents.base_agent import BaseAgent, AgentResult
from config.prompts import DATA_JOURNALIST_SYSTEM_PROMPT, DATA_JOURNALIST_SUMMARY_PROMPT

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


class CompanyLookupService:
    """Stub for company lookup service."""

    def __init__(self, use_mock: bool = False):
        self.news_api = MockCompanyLookupAPI() if use_mock else CompanyLookupAPI()

    @lru_cache(maxsize=100)
    def get_company_name(self, ticker: str) -> str:
        return self.news_api.get_company_name(ticker)


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

    @field_validator("published_at", mode="before")
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

    def get_company_news(
        self, company_name: str, ticker: str, days_back: int, max_articles: int
    ) -> List[Dict[str, Any]]:
        return self.news_api.get_company_news(
            company_name, ticker, days_back, max_articles
        )


class SentimentAnalyzer:
    def __init__(self):
        try:
            from transformers import pipeline

            self._analyzer = pipeline(
                "sentiment-analysis",
                model="yiyanghkust/finbert-tone",
                tokenizer="yiyanghkust/finbert-tone",
            )
        except ImportError:
            logger.warning("Transformers not installed. Using fallback analyzer.")
            self._analyzer = None

    def analyze_text(self, text: str) -> Tuple[float, float]:
        if not text or self._analyzer is None:
            return 0.0, 0.5
        try:
            result = self._analyzer(text[:512])[0]
            label = result["label"].lower()
            confidence = result["score"]
            if "positive" in label:
                return confidence, confidence
            elif "negative" in label:
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
            from bertopic.representation import KeyBERTInspired

            representation_model = KeyBERTInspired()
            self._topic_model = BERTopic(
                verbose=True,
                representation_model=representation_model,
                nr_topics="auto",
                min_topic_size=5,
            )
            self._is_initialized = False
        except ImportError:
            logger.warning("BERTopic not installed. Using simple keyword extraction.")
            self._topic_model = None

    def extract_topics(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        if self._topic_model and len(texts) >= self._topic_model.min_topic_size:
            try:
                if not self._is_initialized:
                    topics, _ = self._topic_model.fit_transform(texts)
                    self._is_initialized = True
                else:
                    topics, _ = self._topic_model.transform(texts)
                topic_info = self._topic_model.get_topic_info()
                meaningful_topics = [
                    info["Name"]
                    for _, info in topic_info.iterrows()
                    if info["Topic"] != -1 and info["Name"] != ""
                ]

                cleaned_topics = []
                for topic_name in meaningful_topics:
                    cleaned_name = re.sub(r"^-?\d+_", "", topic_name)
                    cleaned_name = cleaned_name.replace("_", " ").strip()
                    if cleaned_name:
                        cleaned_topics.append(cleaned_name)

                return cleaned_topics[:5]
            except Exception as e:
                logger.warning(f"BERTopic topic modeling failed: {e}")
                return self._fallback_topic_extraction(texts)
        else:
            if self._topic_model and len(texts) < self._topic_model.min_topic_size:
                logger.warning(
                    f"Not enough texts ({len(texts)}) for BERTopic (min_topic_size={self._topic_model.min_topic_size}). Falling back to keyword extraction."
                )
            return self._fallback_topic_extraction(texts)

    def _fallback_topic_extraction(self, texts: List[str]) -> List[str]:
        all_words = []
        for text in texts:
            all_words.extend(re.findall(r"\b[A-Z][a-z]+\b", text))
            all_words.extend(
                re.findall(
                    r"\b(?:stock|market|earnings|product|launch|investment|analyst|share|revenue)\b",
                    text,
                    re.IGNORECASE,
                )
            )

        filtered_words = [
            word
            for word in all_words
            if word.lower() not in map(str.lower, stop_words) and len(word) > 2
        ]
        counts = Counter(filtered_words)
        return [w for w, _ in counts.most_common(5)]


class DataJournalistAgent(BaseAgent):
    def __init__(
        self,
        news_service: Optional[NewsService] = None,
        lookup_service: Optional[CompanyLookupService] = None,
    ):
        super().__init__("Data Journalist")
        self.news_service = news_service or NewsService()
        self.lookup_service = lookup_service or CompanyLookupService()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        logger.info("DataJournalistAgent initialized.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    def analyze(self, ticker: str, **kwargs) -> AgentResult:
        try:
            ticker = self._validate_ticker(ticker)
            company_name = kwargs.get(
                "company_name"
            ) or self.lookup_service.get_company_name(ticker)
            days_back = kwargs.get("days_back", 7)
            max_articles = kwargs.get("max_articles", 50)
            raw_articles = self._get_cached_news(
                company_name, ticker, days_back, max_articles
            )
            valid_articles = self._validate_articles(raw_articles)

            if not valid_articles:
                return self._create_no_data_result(ticker)

            texts = [f"{a.title}. {a.description}" for a in valid_articles]

            structured_analysis = self._perform_structured_analysis(
                texts, valid_articles
            )

            # Use BaseAgent LLM call
            summary_prompt = DATA_JOURNALIST_SUMMARY_PROMPT.format(
                company_name=company_name,
                ticker=ticker,
                structured_analysis=json.dumps(structured_analysis.dict(), indent=2),
            )
            llm_summary = self._call_llm(summary_prompt, DATA_JOURNALIST_SYSTEM_PROMPT)
            structured_analysis.summary = llm_summary

            overall_confidence = self._calculate_overall_confidence(
                valid_articles, structured_analysis
            )
            metadata = self._prepare_metadata(valid_articles, structured_analysis)

            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis=llm_summary,
                confidence_score=overall_confidence,
                structured_data=structured_analysis.dict(),
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Error in news analysis for {ticker}: {e}")
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis="Analysis failed due to technical error. Please try again later.",
                confidence_score=0.0,
                errors=str(e),
            )

    @lru_cache(maxsize=32)
    def _get_cached_news(
        self, company_name: str, ticker: str, days_back: int, max_articles: int
    ) -> List[Dict[str, Any]]:
        articles = self.news_service.get_company_news(
            company_name, ticker, days_back, max_articles
        )
        logger.info(
            f"Retrieved {len(articles)} raw articles for {ticker} ({company_name})"
        )
        return articles

    def _validate_articles(self, articles: List[Dict[str, Any]]) -> List[NewsArticle]:
        valid = []
        for a in articles:
            try:
                article = NewsArticle(**a)
                if len(article.title) > 15 and article.relevance_score > 0.2:
                    valid.append(article)
            except Exception as e:
                logger.warning(
                    f"Invalid article skipped: {e} - Data: {a.get('title', 'N/A')}"
                )
                continue
        logger.info(f"Validated {len(valid)}/{len(articles)} articles after filtering")
        return valid

    def _perform_structured_analysis(
        self, texts: List[str], articles: List[NewsArticle]
    ) -> NewsAnalysisResult:
        logger.info(f"Performing sentiment & topic analysis on {len(texts)} texts...")
        scores, confs = [], []
        for t in texts:
            s, c = self.sentiment_analyzer.analyze_text(t)
            scores.append(s)
            confs.append(c)

        avg_sentiment_score = float(np.mean(scores)) if scores else 0.0
        avg_sentiment_confidence = float(np.mean(confs)) if confs else 0.0

        overall_sentiment = (
            Sentiment.BULLISH
            if avg_sentiment_score > 0.1
            else Sentiment.BEARISH if avg_sentiment_score < -0.1 else Sentiment.NEUTRAL
        )

        # Topic Modeling
        key_themes = self.topic_modeler.extract_topics(texts)

        # Catalyst Detection
        potential_catalysts = self._detect_catalysts(texts)

        result = NewsAnalysisResult(
            overall_sentiment=overall_sentiment,
            sentiment_score=avg_sentiment_score,
            sentiment_confidence=avg_sentiment_confidence,
            key_themes=key_themes,
            potential_catalysts=potential_catalysts,
            summary="",  # To be added by LLM summary
        )
        logger.info(
            f"Structured analysis complete → Sentiment: {result.overall_sentiment} ({result.sentiment_score:.2f}), "
            f"Themes: {result.key_themes}, Catalysts: {len(result.potential_catalysts)}"
        )
        return result

    def _detect_catalysts(self, texts: List[str]) -> List[Catalyst]:
        catalysts = []
        keywords = {
            FinancialEventType.EARNINGS: [
                "earnings",
                "quarterly results",
                "eps",
                "revenue",
                "financial report",
            ],
            FinancialEventType.PRODUCT_LAUNCH: [
                "launch",
                "new product",
                "announce",
                "release",
                "unveil",
                "innovation",
            ],
            FinancialEventType.MERGER_ACQUISITION: [
                "acquire",
                "merge",
                "takeover",
                "buyout",
                "acquisition",
                "deal",
            ],
            FinancialEventType.REGULATORY: [
                "fda",
                "approval",
                "regulation",
                "investigation",
                "antitrust",
                "compliance",
            ],
            FinancialEventType.EXECUTIVE_CHANGE: [
                "ceo",
                "cfo",
                "resigns",
                "appoints",
                "leadership change",
            ],
            FinancialEventType.GUIDANCE_UPDATE: [
                "guidance",
                "outlook",
                "forecast",
                "expectations",
            ],
            FinancialEventType.PARTNERSHIP: [
                "partnership",
                "collaboration",
                "alliance",
            ],
            FinancialEventType.LAWSUIT: [
                "lawsuit",
                "suit",
                "legal action",
                "settlement",
            ],
        }

        for _, t in enumerate(texts):
            t_lower = t.lower()
            for et, kws in keywords.items():
                if any(k in t_lower for k in kws):
                    snippet_start = max(0, t_lower.find(kws[0]) - 50)
                    snippet_end = min(len(t), snippet_start + 200)
                    description_snippet = (
                        t[snippet_start:snippet_end].strip() + "..."
                        if len(t) > 200
                        else t
                    )

                    catalysts.append(
                        Catalyst(description=description_snippet, event_type=et)
                    )
                    if len(catalysts) >= 5:
                        return catalysts
                    break
        return catalysts

    def _calculate_overall_confidence(
        self, articles: List[NewsArticle], analysis: NewsAnalysisResult
    ) -> float:
        """
        Calculates a single, unified confidence score for the entire Data Journalist analysis.
        This combines data quality, recency, relevance, and NLP model confidence.
        """
        if not articles:
            return 0.1

        data_quantity_score = min(len(articles) / 20.0, 1.0) * 0.25

        recency_score = self._calculate_recency_score(articles) * 0.25

        avg_relevance = (
            np.mean([a.relevance_score for a in articles]) if articles else 0.0
        )
        data_relevance_score = avg_relevance * 0.2

        sentiment_model_confidence = analysis.sentiment_confidence * 0.2

        analysis_richness_score = 0.0
        if analysis.key_themes:
            analysis_richness_score += 0.05
        if analysis.potential_catalysts:
            analysis_richness_score += 0.05

        total_confidence = (
            data_quantity_score
            + recency_score
            + data_relevance_score
            + sentiment_model_confidence
            + analysis_richness_score
        )

        return min(max(total_confidence, 0.1), 0.95)

    def _calculate_recency_score(self, articles: List[NewsArticle]) -> float:
        now = datetime.now(timezone.utc)
        recency_sum = 0.0
        if not articles:
            return 0.0

        for a in articles:
            pub = a.published_at
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)

            days_old = (now - pub).days

            recency_score_article = np.exp(-days_old / 10.0)
            recency_sum += recency_score_article

        return min(recency_sum / len(articles), 1.0)

    def _prepare_metadata(
        self, articles: List[NewsArticle], analysis: NewsAnalysisResult
    ) -> Dict[str, Any]:
        sources = list({a.source for a in articles})

        top_articles_for_display = []
        sorted_articles = sorted(
            articles, key=lambda x: x.relevance_score, reverse=True
        )
        for article in sorted_articles[:5]:
            top_articles_for_display.append(
                {
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "description": article.description,
                    "url": str(article.url) if article.url else None,
                    "relevance_score": article.relevance_score,
                }
            )

        metadata = {
            "articles_analyzed": len(articles),
            "sources_analyzed": sources,
            "avg_relevance_score": np.mean([a.relevance_score for a in articles]),
            "date_range": {
                "oldest": min(a.published_at for a in articles).isoformat(),
                "newest": max(a.published_at for a in articles).isoformat(),
            },
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": "finbert-bertopic-2.0",
            "articles": top_articles_for_display,
        }
        logger.info(
            f"Metadata prepared → {metadata['articles_analyzed']} articles, "
            f"Sources: {', '.join(metadata['sources_analyzed'])}, "
            f"Date range: {metadata['date_range']['oldest']} → {metadata['date_range']['newest']}"
        )
        return metadata

    def _create_no_data_result(self, ticker: str) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            company_ticker=ticker,
            analysis="No recent relevant news articles found for comprehensive analysis. "
            "Consider broadening the search period or checking the ticker symbol.",
            confidence_score=0.05,
            metadata={"articles_found": 0},
            errors="No relevant news data available",
        )
