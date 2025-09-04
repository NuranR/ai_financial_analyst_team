"""News API integration for financial news."""

import requests
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
from config.settings import settings


class NewsAPI:
    """Interface for fetching financial news from various sources."""

    def __init__(self):
        self.api_key = settings.news_api_key
        self.base_url = settings.news_api_base_url
        self.news_relevance_threshold = settings.news_relevance_threshold

    def get_company_news(
        self, company_name: str, ticker: str, days_back: int = 7, max_articles: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch recent news articles about a company.

        Args:
            company_name: Full company name
            ticker: Stock ticker symbol
            days_back: Number of days to look back
            max_articles: Maximum number of articles to return

        Returns:
            List of news articles with metadata
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Search query combining company name and ticker
            query = f'"{company_name}" AND "{ticker}" AND (stock OR shares OR earnings OR financial)'

            params = {
                "q": query,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "sortBy": "relevancy",
                "pageSize": min(max_articles, 50),
                "language": "en",
                "domains": "reuters.com,bloomberg.com,cnbc.com,marketwatch.com,yahoo.com,finance.yahoo.com",
            }

            if self.api_key:
                params["apiKey"] = self.api_key

            response = requests.get(f"{self.base_url}/everything", params=params)
            response.raise_for_status()

            data = response.json()
            articles = data.get("articles", [])

            # Filter and format articles
            formatted_articles = []
            for article in articles[:max_articles]:
                if self._is_relevant_article(article, ticker, company_name):
                    formatted_articles.append(
                        {
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "content": article.get("content", ""),
                            "url": article.get("url", ""),
                            "published_at": article.get("publishedAt", ""),
                            "source": article.get("source", {}).get("name", ""),
                            "relevance_score": self._calculate_relevance(
                                article, ticker, company_name
                            ),
                        }
                    )

            # Sort by relevance score
            formatted_articles.sort(key=lambda x: x["relevance_score"], reverse=True)
            formatted_articles = [
                a
                for a in formatted_articles
                if a["relevance_score"] >= self.news_relevance_threshold
            ]

            logger.info(
                f"Fetched {len(formatted_articles)} relevant articles for {ticker}, and days_back={days_back} max_articles={max_articles}   "
            )
            return formatted_articles

        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            return []

    def _is_relevant_article(
        self, article: Dict, ticker: str, company_name: str
    ) -> bool:
        """Check if article is relevant to the company."""
        text_to_check = (
            f"{article.get('title', '')} {article.get('description', '')}".lower()
        )

        # Must contain ticker or company name
        ticker_match = ticker.lower() in text_to_check
        company_match = company_name.lower() in text_to_check

        # Filter out irrelevant content
        irrelevant_keywords = [
            "recipe",
            "sports",
            "weather",
            "entertainment",
            "celebrity",
        ]
        has_irrelevant = any(
            keyword in text_to_check for keyword in irrelevant_keywords
        )

        return (ticker_match or company_match) and not has_irrelevant

    def _calculate_relevance(
        self, article: Dict, ticker: str, company_name: str
    ) -> float:
        """Calculate relevance score for article."""
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        score = 0.0

        # Base score for containing ticker/company
        if ticker.lower() in text:
            score += 0.5
        if company_name.lower() in text:
            score += 0.3

        # Bonus for financial keywords
        financial_keywords = [
            "earnings",
            "revenue",
            "profit",
            "stock",
            "shares",
            "market",
            "analyst",
            "forecast",
        ]
        for keyword in financial_keywords:
            if keyword in text:
                score += 0.1

        # Bonus for recent articles
        try:
            pub_date = datetime.fromisoformat(
                article.get("publishedAt", "").replace("Z", "+00:00")
            )
            days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days
            if days_old <= 1:
                score += 0.2
            elif days_old <= 3:
                score += 0.1
        except:
            pass

        return min(score, 1.0)


# Fallback for when NewsAPI is not available
class MockNewsAPI:
    """Mock news API for development/testing."""

    def get_company_news(
        self, company_name: str, ticker: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Return mock news data."""
        return [
            {
                "title": f"{company_name} Reports Strong Q4 Earnings",
                "description": f"{company_name} ({ticker}) exceeded analyst expectations with strong quarterly results.",
                "content": f"Sample financial news content about {company_name} performance...",
                "url": "https://example.com/news1",
                "published_at": datetime.now().isoformat(),
                "source": "Mock Financial News",
                "relevance_score": 0.9,
            },
            {
                "title": f"Analysts Upgrade {ticker} Stock Rating",
                "description": f"Major investment firm upgrades {ticker} from Hold to Buy.",
                "content": f"Sample analyst upgrade news for {company_name}...",
                "url": "https://example.com/news2",
                "published_at": (datetime.now() - timedelta(hours=6)).isoformat(),
                "source": "Mock Market Watch",
                "relevance_score": 0.8,
            },
        ]
