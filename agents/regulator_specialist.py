"""Regulator Specialist Agent for SEC filings and compliance analysis."""

from typing import Dict, Any
from datetime import datetime
from loguru import logger

from agents.base_agent import BaseAgent, AgentResult
from data.sec_filings import SECFilingsFetcher
from config.prompts import REGULATOR_SPECIALIST_SYSTEM_PROMPT, REGULATOR_SPECIALIST_ANALYSIS_PROMPT
from config.settings import settings


class RegulatorSpecialistAgent(BaseAgent):
    """
    Agent specializing in regulatory analysis and SEC filings review.
    
    Responsibilities:
    - Analyze SEC filings (10-K, 10-Q, 8-K)
    - Assess regulatory compliance status
    - Identify risk factors from official disclosures
    - Evaluate corporate governance practices
    """
    
    def __init__(self):
        super().__init__("Regulator Specialist")
        self.sec_fetcher = SECFilingsFetcher()
    
    def analyze(self, ticker: str, **kwargs) -> AgentResult:
        """
        Perform regulatory analysis on SEC filings.
        
        Args:
            ticker: Stock ticker symbol
            filing_types: List of filing types to analyze (default: ['10-K', '10-Q', '8-K'])
            max_filings: Maximum number of filings to analyze (default: 5)
            
        Returns:
            AgentResult with regulatory analysis
        """
        try:
            # Validate inputs
            ticker = self._validate_ticker(ticker)
            filing_types = kwargs.get('filing_types', ['10-K', '10-Q', '8-K'])
            max_filings = kwargs.get('max_filings', 5)
            
            logger.info(f"Starting regulatory analysis for {ticker}")
            
            # Fetch SEC filings
            filing_data = self.sec_fetcher.get_company_filings(
                ticker=ticker,
                filing_types=filing_types,
                limit=max_filings
            )
            
            # Extract insights from filings
            filing_insights = self.sec_fetcher.extract_filing_insights(filing_data)
            
            # Prepare data for LLM analysis
            analysis_data = self._format_filing_data_for_analysis(filing_data, filing_insights)
            
            # Generate analysis using LLM
            analysis_prompt = REGULATOR_SPECIALIST_ANALYSIS_PROMPT.format(
                company_name=filing_data['company_info']['name'],
                ticker=ticker,
                filing_type=filing_insights['filing_summary'].get('latest_filing', 'Various'),
                filing_date=filing_insights['filing_summary'].get('latest_filing_date', 'Unknown'),
                financial_data=analysis_data['financial_summary'],
                risk_factors=analysis_data['risk_summary'],
                md_and_a=analysis_data['compliance_summary']
            )
            
            analysis = self._call_llm(analysis_prompt, REGULATOR_SPECIALIST_SYSTEM_PROMPT)
            
            # Calculate confidence based on data availability and recency
            confidence = self._calculate_confidence_score(
                analysis,
                data_quality=self._assess_filing_data_quality(filing_data),
                compliance_health=self._assess_compliance_score(filing_insights),
                data_recency=self._assess_data_recency(filing_data)
            )
            
            metadata = {
                'filings_analyzed': filing_data['total_filings'],
                'filing_types': filing_insights['filing_summary']['filing_types'],
                'latest_filing': filing_insights['filing_summary']['latest_filing'],
                'compliance_status': filing_insights['compliance_assessment']['status'],
                'regulatory_health': filing_insights['regulatory_health'],
                'risk_signals': filing_insights['compliance_assessment']['risk_signals'],
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Completed regulatory analysis for {ticker}")
            
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis=analysis,
                confidence_score=confidence,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error in regulatory analysis for {ticker}: {str(e)}")
            return AgentResult(
                agent_name=self.name,
                company_ticker=ticker,
                analysis="",
                confidence_score=0.0,
                errors=str(e)
            )
    
    def _format_filing_data_for_analysis(self, filing_data: Dict[str, Any], 
                                       filing_insights: Dict[str, Any]) -> Dict[str, str]:
        """Format filing data for LLM analysis."""
        
        # Financial summary
        company_info = filing_data['company_info']
        financial_summary = f"""
Company: {company_info['name']}
Industry: {company_info.get('sicDescription', 'Unknown')}
Entity Type: {company_info.get('entityType', 'Unknown')}
Filer Category: {company_info.get('category', 'Unknown')}
"""
        
        # Risk factors summary
        compliance_data = filing_insights.get('compliance_assessment', {})
        risk_summary = f"""
Compliance Status: {compliance_data.get('status', 'Unknown')}
Risk Signals Detected: {len(compliance_data.get('risk_signals', []))}

Risk Signals:
"""
        for signal in compliance_data.get('risk_signals', []):
            risk_summary += f"• {signal}\n"
        
        if not compliance_data.get('risk_signals'):
            risk_summary += "• No significant risk signals detected\n"
        
        # Filing compliance summary
        filing_summary = filing_insights.get('filing_summary', {})
        compliance_summary = f"""
Total Filings Available: {filing_summary.get('total_filings', 0)}
Filing Types: {', '.join(filing_summary.get('filing_types', []))}
Latest Filing: {filing_summary.get('latest_filing', 'None')} ({filing_summary.get('latest_filing_date', 'Unknown')})
Regulatory Health: {filing_insights.get('regulatory_health', 'Unknown')}

Recent Filings:
"""
        
        for filing in filing_data.get('filings', [])[:3]:  # Show last 3 filings
            compliance_summary += f"• {filing['form_type']} - {filing['filing_date']} - {filing['description']}\n"
        
        return {
            'financial_summary': financial_summary,
            'risk_summary': risk_summary,
            'compliance_summary': compliance_summary
        }
    
    def _assess_filing_data_quality(self, filing_data: Dict[str, Any]) -> float:
        """Assess the quality of filing data."""
        score = 0.5  # Base score
        
        # More filings = better data quality
        filing_count = filing_data.get('total_filings', 0)
        if filing_count >= 3:
            score += 0.2
        elif filing_count >= 1:
            score += 0.1
        
        # Variety of filing types = better coverage
        filings = filing_data.get('filings', [])
        filing_types = set(f['form_type'] for f in filings)
        if '10-K' in filing_types:
            score += 0.1  # Annual report is important
        if '10-Q' in filing_types:
            score += 0.1  # Quarterly reports are important
        
        # Check if this is mock data
        if filing_data.get('note') and 'mock' in filing_data['note'].lower():
            score = max(score - 0.3, 0.1)  # Reduce score for mock data
        
        return min(score, 1.0)
    
    def _assess_compliance_score(self, filing_insights: Dict[str, Any]) -> float:
        """Assess compliance health score."""
        compliance_data = filing_insights.get('compliance_assessment', {})
        health = filing_insights.get('regulatory_health', 'Unknown').lower()
        
        health_scores = {
            'good': 0.3,
            'fair': 0.1,
            'concerning': -0.2,
            'unknown': 0.0
        }
        
        base_score = health_scores.get(health, 0.0)
        
        # Penalty for risk signals
        risk_count = len(compliance_data.get('risk_signals', []))
        penalty = risk_count * 0.1
        
        return max(base_score - penalty, -0.3)
    
    def _assess_data_recency(self, filing_data: Dict[str, Any]) -> float:
        """Assess how recent the filing data is."""
        try:
            filings = filing_data.get('filings', [])
            if not filings:
                return 0.0
            
            latest_filing = filings[0]
            filing_date = datetime.fromisoformat(latest_filing['filing_date'])
            days_old = (datetime.now() - filing_date).days
            
            if days_old <= 90:  # Within 3 months
                return 0.2
            elif days_old <= 180:  # Within 6 months
                return 0.1
            else:
                return 0.0
                
        except Exception:
            return 0.0
