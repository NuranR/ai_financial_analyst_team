"""Data Journalist Agent for analyzing financial news and social media."""

from typing import Dict, Any, List
from datetime import datetime
from loguru import logger

from agents.base_agent import BaseAgent, AgentResult
from api.news_api import NewsAPI, MockNewsAPI
from config.prompts import DATA_JOURNALIST_SYSTEM_PROMPT, DATA_JOURNALIST_ANALYSIS_PROMPT
from config.settings import settings


class DataJournalistAgent(BaseAgent):
    """
    Agent specializing in financial news analysis and sentiment extraction.
    
    Responsibilities:
    - Fetch and analyze recent news articles
    - Extract sentiment and market impact
    - Identify key themes and events
    - Assess potential stock price catalysts
    """
    
    def __init__(self):
        super().__init__("Data Journalist")
        
        # Initialize news API (use mock if no API key)
        if settings.news_api_key:
            self.news_api = NewsAPI()
            logger.info("Using live NewsAPI")
        else:
            self.news_api = MockNewsAPI()
            logger.warning("Using mock news data - set NEWS_API_KEY for live data")
    
    def analyze(self, ticker: str, **kwargs) -> AgentResult:
        """
        Analyze financial news and sentiment for a company.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Optional company name for better search
            days_back: Number of days to analyze (default: 7)
            max_articles: Maximum articles to analyze (default: 20)
            
        Returns:
            AgentResult with news analysis and sentiment
        """
        try:
            # Validate and prepare inputs
            ticker = self._validate_ticker(ticker)
            company_name = kwargs.get('company_name', self._get_company_name(ticker))
            days_back = kwargs.get('days_back', 7)
            max_articles = kwargs.get('max_articles', 20)
            
            logger.info(f"Starting news analysis for {ticker} ({company_name})")
            
            # Fetch news articles
            articles = self.news_api.get_company_news(
                company_name=company_name,
                ticker=ticker,
                days_back=days_back,
                max_articles=max_articles
            )
            
            if not articles:
                return AgentResult(
                    agent_name=self.name,
                    company_ticker=ticker,
                    analysis="No recent news articles found for analysis.",
                    confidence_score=0.1,
                    metadata={'articles_found': 0},
                    errors="No news data available"
                )
            
            # Prepare content for LLM analysis
            news_content = self._format_news_for_analysis(articles)
            
            # Generate analysis using LLM
            analysis_prompt = DATA_JOURNALIST_ANALYSIS_PROMPT.format(
                company_name=company_name,
                ticker=ticker,
                news_content=news_content
            )
            
            analysis = self._call_llm(analysis_prompt, DATA_JOURNALIST_SYSTEM_PROMPT)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence_score(
                analysis,
                data_quality=len(articles) / max_articles,
                recency=self._calculate_recency_score(articles),
                relevance=self._calculate_average_relevance(articles)
            )
            
            metadata = {
                'articles_analyzed': len(articles),
                'date_range': f"{days_back} days",
                'top_sources': self._get_top_sources(articles),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Completed news analysis for {ticker}")
            
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis=analysis,
                confidence_score=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in news analysis for {ticker}: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis="",
                confidence_score=0.0,
                errors=str(e)
            )
    
    def _format_news_for_analysis(self, articles: List[Dict[str, Any]]) -> str:
        """Format news articles for LLM analysis."""
        formatted_content = []
        
        logger.info(f"📰 Formatting {len(articles)} articles for analysis")
        print(f"📰 FORMATTING {len(articles)} ARTICLES FOR ANALYSIS")
        
        for i, article in enumerate(articles[:10], 1):  # Limit to top 10 for token efficiency
            content = f"""
Article {i}:
Title: {article['title']}
Source: {article['source']}
Published: {article['published_at'][:10]}  # Just date
Description: {article['description']}
Relevance Score: {article['relevance_score']:.2f}
---"""
            formatted_content.append(content)
            
            # Log each article being formatted
            logger.info(f"📋 Article {i}: {article['title'][:100]}...")
            print(f"📋 ARTICLE {i}: {article['title']}")
        
        final_content = "\n".join(formatted_content)
        content_length = len(final_content)
        
        logger.info(f"📝 Formatted content length: {content_length} characters")
        logger.info(f"📝 Content preview: {final_content[:500]}...")
        
        print(f"📝 FORMATTED CONTENT LENGTH: {content_length} CHARACTERS")
        print(f"📝 FULL FORMATTED CONTENT FOR DISTILBART:")
        print(f"{final_content}")
        print(f"📝 END OF FORMATTED CONTENT")
        
        return final_content
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker (simplified mapping)."""
        # This could be enhanced with a proper API or database lookup
        ticker_to_name = {
            'AAPL': 'Apple Inc.',
            'GOOGL': 'Alphabet Inc.',
            'GOOG': 'Alphabet Inc.',
            'MSFT': 'Microsoft Corporation',
            'TSLA': 'Tesla Inc.',
            'AMZN': 'Amazon.com Inc.',
            'META': 'Meta Platforms Inc.',
            'NFLX': 'Netflix Inc.',
            'NVDA': 'NVIDIA Corporation',
            'AMD': 'Advanced Micro Devices Inc.'
        }
        return ticker_to_name.get(ticker, f"{ticker} Corporation")
    
    def _calculate_recency_score(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate how recent the articles are."""
        if not articles:
            return 0.0
        
        try:
            recent_count = 0
            for article in articles:
                pub_date = datetime.fromisoformat(article['published_at'].replace('Z', '+00:00'))
                days_old = (datetime.now() - pub_date.replace(tzinfo=None)).days
                if days_old <= 2:
                    recent_count += 1
            
            return recent_count / len(articles)
        except:
            return 0.5  # Default score if date parsing fails
    
    def _calculate_average_relevance(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate average relevance score of articles."""
        if not articles:
            return 0.0
        
        total_relevance = sum(article['relevance_score'] for article in articles)
        return total_relevance / len(articles)
    
    def _get_top_sources(self, articles: List[Dict[str, Any]]) -> List[str]:
        """Get list of top news sources."""
        sources = [article['source'] for article in articles if article['source']]
        unique_sources = list(set(sources))
        return unique_sources[:5]  # Top 5 unique sources
