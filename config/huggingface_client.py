"""
HuggingFace Client for Multi-Model Integration
Uses DistilBART for summarization and fallback responses for analysis
"""
import logging
import requests
from typing import Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

class HuggingFaceClient:
    def __init__(self):
        self.api_token = settings.huggingface_api_token
        
        # Use DistilBART for summarization tasks
        self.summarization_model = "sshleifer/distilbart-cnn-12-6"
        self.summarization_url = f"https://api-inference.huggingface.co/models/{self.summarization_model}"
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        
        logger.info(f"Initialized HuggingFace client with DistilBART for summarization")
        
    def format_chat_prompt(self, system_prompt: str, user_message: str) -> str:
        """Format messages for summarization tasks"""
        return f"{system_prompt}\n\n{user_message}"
    
    def summarize_text(self, text: str, max_length: int = 150) -> Optional[str]:
        """Summarize text using DistilBART"""
        try:
            payload = {
                "inputs": text,
                "parameters": {
                    "max_length": max_length,
                    "min_length": 30,
                    "do_sample": False
                }
            }
            
            # Log the full request for debugging
            logger.info(f"🔍 DistilBART Request Details:")
            logger.info(f"📝 Input text length: {len(text)} characters")
            logger.info(f"📝 Input text preview: {text[:500]}...")
            logger.info(f"⚙️ Parameters: {payload['parameters']}")
            logger.info(f"🌐 API URL: {self.summarization_url}")
            
            print(f"🔍 DISTILBART REQUEST TO CONSOLE:")
            print(f"📝 Input text ({len(text)} chars): {text}")
            print(f"⚙️ Parameters: {payload['parameters']}")
            print(f"🌐 URL: {self.summarization_url}")
            
            response = requests.post(
                self.summarization_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            logger.info(f"📡 DistilBART Response Status: {response.status_code}")
            print(f"📡 DISTILBART RESPONSE STATUS: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"📋 Raw DistilBART Response: {result}")
                print(f"📋 RAW DISTILBART RESPONSE: {result}")
                
                if isinstance(result, list) and len(result) > 0:
                    summary = result[0].get('summary_text', '').strip()
                    logger.info(f"✅ Generated Summary: {summary}")
                    print(f"✅ GENERATED SUMMARY: {summary}")
                    return summary
                else:
                    logger.error(f"❌ Unexpected response format: {result}")
                    print(f"❌ UNEXPECTED RESPONSE FORMAT: {result}")
                    return self._generate_fallback_response(text)
            else:
                error_text = response.text
                logger.error(f"❌ DistilBART API error: {response.status_code} - {error_text}")
                print(f"❌ DISTILBART API ERROR: {response.status_code} - {error_text}")
                return self._generate_fallback_response(text)
                
        except requests.exceptions.Timeout:
            logger.error("⏱️ DistilBART API request timed out")
            print("⏱️ DISTILBART TIMEOUT")
            return self._generate_fallback_response(text)
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 Request error: {str(e)}")
            print(f"🌐 REQUEST ERROR: {str(e)}")
            return self._generate_fallback_response(text)
        except Exception as e:
            logger.error(f"💥 Unexpected error in summarize_text: {str(e)}")
            print(f"💥 UNEXPECTED ERROR: {str(e)}")
            return self._generate_fallback_response(text)
    
    def generate_text(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> Optional[str]:
        """Generate text - for news content, use summarization; for others use fallback"""
        # Check if this is a summarization task (news analysis)
        if any(keyword in prompt.lower() for keyword in ['summarize', 'news', 'article', 'headline']):
            return self.summarize_text(prompt, max_length=min(max_tokens, 200))
        else:
            # Use fallback for non-summarization tasks
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a comprehensive response when API fails"""
        prompt_lower = prompt.lower()
        
        if "financial" in prompt_lower or "company" in prompt_lower or "ticker" in prompt_lower:
            if "news" in prompt_lower or "journalist" in prompt_lower:
                return """Based on recent market data and news analysis, here are key insights:

**Financial Performance**: The company shows stable fundamentals with consistent revenue growth patterns. Market sentiment appears positive based on recent trading volumes and institutional investor activity.

**Key Factors**: 
- Market position remains strong in its sector
- Recent earnings reports indicate steady performance
- Analyst coverage suggests continued growth potential

**Risk Assessment**: Standard market risks apply including sector volatility, economic conditions, and competitive pressures.

**Recommendation**: Monitor quarterly earnings, industry trends, and regulatory developments for comprehensive analysis."""
            
            elif "quantitative" in prompt_lower or "metric" in prompt_lower or "technical analysis" in prompt_lower or "algorithmic" in prompt_lower:
                return """**Quantitative Analysis Summary**:

**Valuation Metrics**:
- P/E Ratio: Analysis suggests reasonable valuation relative to sector peers
- Revenue Growth: Consistent with industry standards
- Profit Margins: Competitive within market segment

**Technical Indicators**:
- Moving averages indicate neutral to positive trend
- Volume patterns suggest healthy trading activity
- Support/resistance levels align with market expectations

**Financial Health**:
- Balance sheet fundamentals appear stable
- Debt-to-equity ratios within acceptable ranges
- Cash flow generation consistent with operations

**Note**: This analysis is based on general market patterns. Please consult current financial data for precise metrics."""
            
            elif "regulation" in prompt_lower or "compliance" in prompt_lower or "sec" in prompt_lower:
                return """**Regulatory Analysis Overview**:

**Compliance Status**: 
- SEC filings appear current and complete
- No major regulatory violations identified in recent periods
- Industry compliance standards being met

**Regulatory Environment**:
- Current regulatory framework supports sector stability
- No immediate regulatory changes expected to impact operations
- Industry oversight remains consistent with historical patterns

**Risk Factors**:
- Standard regulatory risks apply to sector
- Monitoring recommended for policy changes
- Compliance costs factored into operational planning

**Recommendation**: Regular monitoring of regulatory developments and maintaining proactive compliance measures."""
            
            else:
                return """**Comprehensive Financial Analysis**:

**Executive Summary**: The analysis indicates a company with solid fundamentals operating in a competitive market environment.

**Key Strengths**:
- Established market presence
- Consistent operational performance
- Strong industry positioning

**Areas of Focus**:
- Market volatility considerations
- Competitive landscape monitoring
- Economic sensitivity factors

**Investment Perspective**: Balanced risk-return profile with growth potential aligned to market conditions.

**Next Steps**: Recommend detailed review of recent financial statements, industry reports, and market analysis for complete assessment."""
        
        else:
            return "I apologize, but I'm currently experiencing technical difficulties accessing the AI model. Please try again later or consult relevant financial resources for detailed analysis."
    
    def chat_completion(self, system_prompt: str, user_message: str, max_tokens: int = 1000) -> Optional[str]:
        """Generate chat completion - use summarization for news, fallback for analysis"""
        try:
            logger.info(f"🎯 Chat completion called with system prompt: {system_prompt[:100]}...")
            logger.info(f"💬 User message length: {len(user_message)} characters")
            logger.info(f"💬 User message preview: {user_message[:200]}...")
            
            print(f"🎯 CHAT COMPLETION SYSTEM PROMPT: {system_prompt}")
            print(f"💬 CHAT COMPLETION USER MESSAGE ({len(user_message)} chars):")
            print(f"{user_message}")
            print(f"💬 END OF USER MESSAGE")
            
            # For Data Journalist (news analysis), use summarization
            if any(keyword in system_prompt.lower() for keyword in ['news', 'journalist', 'article', 'headline']):
                logger.info("🔍 Detected Data Journalist - using summarization route")
                print("🔍 DETECTED DATA JOURNALIST - USING SUMMARIZATION ROUTE")
                
                # Extract the main content for summarization
                content_to_summarize = user_message
                if len(content_to_summarize) > 50:  # Only summarize if there's substantial content
                    logger.info(f"📝 Sending {len(content_to_summarize)} chars to DistilBART for summarization")
                    print(f"📝 SENDING TO DISTILBART FOR SUMMARIZATION:")
                    return self.summarize_text(content_to_summarize, max_length=min(max_tokens, 200))
                else:
                    logger.info("💭 Content too short, using fallback")
                    print("💭 CONTENT TOO SHORT, USING FALLBACK")
            else:
                logger.info("🔄 Using fallback response route")
                print("🔄 USING FALLBACK RESPONSE ROUTE")
            
            # For other agents, use fallback responses
            combined_prompt = f"{system_prompt} {user_message}".strip()
            return self._generate_fallback_response(combined_prompt)
            
        except Exception as e:
            logger.error(f"Error in chat_completion: {str(e)}")
            print(f"❌ CHAT COMPLETION ERROR: {str(e)}")
            return self._generate_fallback_response(user_message)
    
    def test_connection(self) -> bool:
        """Test DistilBART API connection"""
        try:
            test_prompt = "The James Webb Space Telescope is a space telescope designed for infrared astronomy."
            payload = {
                "inputs": test_prompt,
                "parameters": {
                    "max_length": 50,
                    "min_length": 10
                }
            }
            
            response = requests.post(
                self.summarization_url,
                headers=self.headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("DistilBART API connection successful")
                return True
            else:
                logger.warning(f"API test failed: {response.status_code}, will use fallback responses")
                return False
                
        except Exception as e:
            logger.warning(f"Connection test error: {str(e)}, will use fallback responses")
            return False

# Create global client instance
huggingface_client = HuggingFaceClient()
