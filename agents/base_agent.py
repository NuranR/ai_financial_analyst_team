"""Base agent class for A-FIN system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger
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
        
        # Initialize HuggingFace client
        self.hf_client = None
        if settings.huggingface_api_token:
            try:
                from config.huggingface_client import huggingface_client
                self.hf_client = huggingface_client
                logger.info(f"Initialized {self.name} with HuggingFace Llama model")
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFace: {e}")
                raise ValueError("HuggingFace client required but failed to initialize")
        else:
            raise ValueError("HUGGINGFACE_API_TOKEN is required")
            
        logger.info(f"Initialized {self.name} agent")
    
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
        Call HuggingFace LLM to generate a response.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            
        Returns:
            str: LLM response
        """
        if not self.hf_client:
            raise ValueError("HuggingFace client not available")
        
        try:
            logger.info(f"🤖 {self.name} calling LLM with prompt length: {len(prompt)} chars")
            logger.info(f"🎯 System prompt: {system_prompt[:100] if system_prompt else 'None'}...")
            logger.info(f"💬 User prompt preview: {prompt[:300]}...")
            
            print(f"🤖 {self.name.upper()} CALLING LLM:")
            print(f"📏 PROMPT LENGTH: {len(prompt)} characters")
            print(f"🎯 SYSTEM PROMPT: {system_prompt}")
            print(f"💬 FULL USER PROMPT:")
            print(f"{prompt}")
            print(f"💬 END OF PROMPT")
            
            if system_prompt:
                response = self.hf_client.chat_completion(
                    system_prompt=system_prompt,
                    user_message=prompt,
                    max_tokens=settings.max_tokens
                )
            else:
                response = self.hf_client.generate_text(
                    prompt=prompt,
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature
                )
            
            logger.info(f"✅ {self.name} received LLM response length: {len(response) if response else 0} chars")
            print(f"✅ {self.name.upper()} LLM RESPONSE LENGTH: {len(response) if response else 0} CHARS")
            
            if response:
                return response
            else:
                # Use fallback with combined context
                combined_prompt = f"{system_prompt or ''} {prompt}".strip()
                logger.info(f"⚠️ {self.name} using fallback response")
                print(f"⚠️ {self.name.upper()} USING FALLBACK RESPONSE")
                return self.hf_client._generate_fallback_response(combined_prompt)
                
        except Exception as e:
            logger.error(f"HuggingFace error in {self.name}: {str(e)}")
            print(f"❌ {self.name.upper()} HUGGINGFACE ERROR: {str(e)}")
            # Use fallback with combined context
            combined_prompt = f"{system_prompt or ''} {prompt}".strip()
            return self.hf_client._generate_fallback_response(combined_prompt)
    
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
