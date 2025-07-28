"""Base agent class for A-FIN system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger
import openai
from config.settings import settings


class AgentResult(BaseModel):
    """Standard result format for all agents."""
    agent_name: str
    company_ticker: str
    analysis: str
    confidence_score: float
    metadata: Dict[str, Any] = {}
    errors: Optional[str] = None


class BaseAgent(ABC):
    """Base class for all financial analysis agents."""
    
    def __init__(self, name: str, model: str = None):
        self.name = name
        self.model = model or settings.default_llm_model
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        logger.info(f"Initialized {self.name} agent with model {self.model}")
    
    @abstractmethod
    def analyze(self, ticker: str, **kwargs) -> AgentResult:
        """
        Perform analysis for the given company ticker.
        
        Args:
            ticker: Company stock ticker symbol
            **kwargs: Additional parameters specific to each agent
            
        Returns:
            AgentResult: Standardized analysis result
        """
        pass
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """
        Call the language model with the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            str: LLM response
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error calling LLM in {self.name}: {str(e)}")
            raise
    
    def _validate_ticker(self, ticker: str) -> str:
        """
        Validate and normalize ticker symbol.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            str: Normalized ticker
        """
        if not ticker or not isinstance(ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        
        return ticker.upper().strip()
    
    def _calculate_confidence_score(self, analysis: str, **factors) -> float:
        """
        Calculate confidence score for the analysis.
        
        Args:
            analysis: The analysis text
            **factors: Additional factors to consider
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Base implementation - can be overridden by specific agents
        base_score = 0.7
        
        # Adjust based on analysis length (more comprehensive = higher confidence)
        if len(analysis) > 1000:
            base_score += 0.1
        elif len(analysis) < 300:
            base_score -= 0.1
        
        # Adjust based on specific factors
        for factor, value in factors.items():
            if factor == "data_quality" and isinstance(value, (int, float)):
                base_score += (value - 0.5) * 0.2
        
        return max(0.0, min(1.0, base_score))
