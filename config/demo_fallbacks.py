"""Demo fallback responses for A-FIN system."""

# Pre-computed analysis responses for demo purposes
DEMO_RESPONSES = {
    "AAPL": {
        "data_journalist": """Apple shows strong positive sentiment in recent news. Key highlights: iPhone 16 launch driving revenue growth, Services segment expansion, and strong quarterly earnings. Market confidence remains high with analysts upgrading price targets. Sentiment: BULLISH.""",
        
        "quantitative_analyst": """AAPL technical analysis shows strong momentum. Current price $214.64 represents 15.2% gain over 6-month period. RSI at 58 indicates healthy momentum without overbought conditions. Moving averages suggest upward trend continuation. Volatility: MODERATE, Trend: BULLISH.""",
        
        "regulator_specialist": """Apple's regulatory position remains stable. Recent 10-K filing shows strong compliance framework and manageable regulatory risks. No significant regulatory red flags identified. ESG commitments align with regulatory expectations. Risk Level: LOW.""",
        
        "lead_analyst": """INVESTMENT RECOMMENDATION: BUY. Apple demonstrates strong fundamentals across news sentiment, technical indicators, and regulatory compliance. Price target: $230 (7.2% upside). Risk-adjusted return favorable with moderate volatility. Suitable for growth-oriented portfolios."""
    },
    
    "TSLA": {
        "data_journalist": """Tesla news cycle remains dynamic with mixed sentiment. Positive: FSD improvements and Cybertruck production ramp. Concerns: China market competition and regulatory scrutiny. Overall sentiment: CAUTIOUSLY OPTIMISTIC.""",
        
        "quantitative_analyst": """TSLA shows high volatility with 34% price swing over 6 months. Current technical indicators mixed - support at $180, resistance at $220. High beta indicates amplified market moves. Volatility: HIGH, Trend: SIDEWAYS.""",
        
        "regulator_specialist": """Tesla faces moderate regulatory headwinds. NHTSA investigations ongoing for FSD. SEC oversight on production guidance. China regulatory environment challenging. Risk Level: MODERATE-HIGH.""",
        
        "lead_analyst": """INVESTMENT RECOMMENDATION: HOLD. Tesla presents high-risk, high-reward profile. Strong innovation pipeline offset by regulatory uncertainties and market volatility. Suitable for risk-tolerant investors only."""
    },
    
    "MSFT": {
        "data_journalist": """Microsoft demonstrates consistent positive news flow. AI integration across products driving growth narrative. Cloud market share expansion and enterprise adoption accelerating. Strong leadership positioning in AI race. Sentiment: BULLISH.""",
        
        "quantitative_analyst": """MSFT shows steady upward momentum with low volatility. 12% gain over 6 months with consistent support levels. Strong dividend yield provides downside protection. Technical indicators strongly positive. Volatility: LOW, Trend: BULLISH.""",
        
        "regulator_specialist": """Microsoft maintains strong regulatory compliance. Proactive approach to AI governance and data privacy. Minimal regulatory risks identified. Strong ESG positioning enhances regulatory standing. Risk Level: LOW.""",
        
        "lead_analyst": """INVESTMENT RECOMMENDATION: STRONG BUY. Microsoft combines growth potential with defensive characteristics. AI leadership position provides competitive moat. Excellent risk-adjusted returns. Suitable for all investor profiles."""
    }
}

DEMO_METADATA = {
    "AAPL": {
        "confidence_scores": {"data_journalist": 88, "quantitative_analyst": 85, "regulator_specialist": 90, "lead_analyst": 87},
        "articles_count": 15,
        "current_price": 214.64,
        "filings_count": 8,
        "recommendation": "BUY"
    },
    "TSLA": {
        "confidence_scores": {"data_journalist": 72, "quantitative_analyst": 68, "regulator_specialist": 65, "lead_analyst": 70},
        "articles_count": 22,
        "current_price": 187.35,
        "filings_count": 6,
        "recommendation": "HOLD"
    },
    "MSFT": {
        "confidence_scores": {"data_journalist": 92, "quantitative_analyst": 89, "regulator_specialist": 94, "lead_analyst": 91},
        "articles_count": 18,
        "current_price": 423.15,
        "filings_count": 7,
        "recommendation": "STRONG BUY"
    }
}
