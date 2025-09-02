"""Lead Analyst Agent - The orchestrator that synthesizes all insights."""

from typing import Dict, Any, List
from datetime import datetime
from loguru import logger

from agents.base_agent import BaseAgent, AgentResult
from agents.data_journalist import DataJournalistAgent
from agents.quantitative_analyst import QuantitativeAnalystAgent
from agents.regulator_specialist import RegulatorSpecialistAgent
from config.prompts import LEAD_ANALYST_SYSTEM_PROMPT, LEAD_ANALYST_SYNTHESIS_PROMPT
from config.settings import settings


class LeadAnalystAgent(BaseAgent):
    """
    Lead Analyst Agent - Orchestrates and synthesizes all analysis.
    
    Responsibilities:
    - Coordinate analysis from all other agents
    - Synthesize insights into comprehensive investment brief
    - Provide final investment recommendation with rationale
    - Calculate risk-adjusted recommendations
    """
    
    def __init__(self):
        super().__init__("Lead Analyst")
        
        # Initialize all other agents
        self.data_journalist = DataJournalistAgent()
        self.quant_analyst = QuantitativeAnalystAgent()
        self.regulator_specialist = RegulatorSpecialistAgent()
        
        logger.info("Lead Analyst initialized with all sub-agents")
    
    def analyze(self, ticker: str, **kwargs) -> AgentResult:
        """
        Perform comprehensive analysis by orchestrating all agents.
        
        Args:
            ticker: Stock ticker symbol
            enable_news: Whether to run news analysis (default: True)
            enable_quant: Whether to run quantitative analysis (default: True)
            enable_regulatory: Whether to run regulatory analysis (default: True)
            analysis_period: Time period for analysis (default: 3mo)
            
        Returns:
            AgentResult with comprehensive investment brief
        """
        try:
            # Validate inputs
            ticker = self._validate_ticker(ticker)
            enable_news = kwargs.get('enable_news', True)
            enable_quant = kwargs.get('enable_quant', True)
            enable_regulatory = kwargs.get('enable_regulatory', True)
            analysis_period = kwargs.get('analysis_period', '3mo')
            
            logger.info(f"Starting comprehensive analysis for {ticker}")
            
            # Run individual agent analyses
            agent_results = {}
            
            # 1. News Analysis
            if enable_news:
                logger.info("Running news analysis...")
                news_result = self.data_journalist.analyze(ticker, days_back=14, max_articles=20)
                agent_results['news'] = news_result
            
            # 2. Quantitative Analysis  
            if enable_quant:
                logger.info("Running quantitative analysis...")
                quant_result = self.quant_analyst.analyze(ticker, period=analysis_period)
                agent_results['quant'] = quant_result
            
            # 3. Regulatory Analysis
            if enable_regulatory:
                logger.info("Running regulatory analysis...")
                reg_result = self.regulator_specialist.analyze(ticker, max_filings=3)
                agent_results['regulatory'] = reg_result
            
            # Extract key data for synthesis
            current_price, market_cap = self._extract_financial_metrics(agent_results)
            
            # Generate comprehensive analysis
            synthesis_prompt = LEAD_ANALYST_SYNTHESIS_PROMPT.format(
                company_name=self._get_company_name(ticker),
                ticker=ticker,
                news_analysis=self._format_agent_result(agent_results.get('news')),
                quant_analysis=self._format_agent_result(agent_results.get('quant')),
                regulatory_analysis=self._format_agent_result(agent_results.get('regulatory')),
                current_price=current_price,
                market_cap=market_cap
            )
            
            comprehensive_analysis = self._call_llm(synthesis_prompt, LEAD_ANALYST_SYSTEM_PROMPT)
            
            # Calculate overall confidence score
            confidence = self._calculate_overall_confidence(agent_results)
            
            # Prepare metadata
            metadata = {
                'agents_run': list(agent_results.keys()),
                'individual_confidences': {k: v.confidence_score for k, v in agent_results.items() if v},
                'current_price': current_price,
                'market_cap': market_cap,
                'analysis_period': analysis_period,
                'analysis_timestamp': datetime.now().isoformat(),
                'recommendation_basis': self._extract_recommendation_basis(agent_results)
            }
            
            logger.info(f"Completed comprehensive analysis for {ticker}")
            
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis=comprehensive_analysis,
                confidence_score=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {ticker}: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis="",
                confidence_score=0.0,
                errors=str(e)
            )
    
    def _format_agent_result(self, result: AgentResult) -> str:
        """Format individual agent result for synthesis."""
        if not result or result.errors:
            return "Analysis not available or failed."
        
        return f"""
Agent: {result.agent_name}
Confidence: {result.confidence_score:.2f}

Analysis:
{result.analysis}

Key Metadata: {result.metadata}
"""
    
    def _extract_financial_metrics(self, agent_results: Dict[str, AgentResult]) -> tuple:
        """Extract current price and market cap from agent results."""
        current_price = "Unknown"
        market_cap = "Unknown"
        
        # Try to get from quantitative analysis first
        quant_result = agent_results.get('quant')
        if quant_result and not quant_result.errors:
            current_price = quant_result.metadata.get('current_price', 'Unknown')
            if current_price != 'Unknown':
                current_price = f"{current_price:.2f}"
        
        return current_price, market_cap
    
    def _calculate_overall_confidence(self, agent_results: Dict[str, AgentResult]) -> float:
        """Calculate overall confidence based on individual agent results."""
        confidences = []
        weights = {'news': 0.2, 'quant': 0.5, 'regulatory': 0.3}  # Quant analysis weighted most
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for agent_type, result in agent_results.items():
            if result and not result.errors:
                weight = weights.get(agent_type, 0.33)
                weighted_score += result.confidence_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # Bonus for having multiple successful analyses
        completeness_bonus = len([r for r in agent_results.values() if r and not r.errors]) * 0.05
        
        final_score = (weighted_score / total_weight) + completeness_bonus
        return min(final_score, 1.0)
    
    def _extract_recommendation_basis(self, agent_results: Dict[str, AgentResult]) -> Dict[str, str]:
        """Extract key factors that influence the recommendation."""
        basis = {}
        
        # News sentiment
        news_result = agent_results.get('news')
        if news_result and not news_result.errors:
            basis['news_sentiment'] = 'Positive' if 'positive' in news_result.analysis.lower() else 'Mixed'
        
        # Quantitative signals
        quant_result = agent_results.get('quant')
        if quant_result and not quant_result.errors:
            if 'bullish' in quant_result.analysis.lower():
                basis['technical_outlook'] = 'Bullish'
            elif 'bearish' in quant_result.analysis.lower():
                basis['technical_outlook'] = 'Bearish'
            else:
                basis['technical_outlook'] = 'Neutral'
        
        # Regulatory health
        reg_result = agent_results.get('regulatory')
        if reg_result and not reg_result.errors:
            basis['regulatory_health'] = reg_result.metadata.get('regulatory_health', 'Unknown')
        
        return basis
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker."""
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
