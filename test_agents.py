"""Test the completed agents."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.data_journalist import DataJournalistAgent
from agents.quantitative_analyst import QuantitativeAnalystAgent

def test_agents():
    print("🧪 Testing A-FIN Agents...\n")
    
    # Test Data Journalist
    print("📰 Testing Data Journalist Agent...")
    journalist = DataJournalistAgent()
    news_result = journalist.analyze("AAPL", company_name="Apple Inc.")
    
    if news_result.errors:
        print(f"❌ Data Journalist Error: {news_result.errors}")
    else:
        print(f"✅ Data Journalist Success!")
        print(f"   Articles analyzed: {news_result.metadata.get('articles_analyzed', 0)}")
        print(f"   Confidence: {news_result.confidence_score:.2f}")
        print(f"   Analysis preview: {news_result.analysis[:200]}...")
    
    print("\n" + "="*50 + "\n")
    
    # Test Quantitative Analyst
    print("📈 Testing Quantitative Analyst Agent...")
    quant = QuantitativeAnalystAgent()
    quant_result = quant.analyze("AAPL", period="3mo")
    
    if quant_result.errors:
        print(f"❌ Quantitative Analyst Error: {quant_result.errors}")
    else:
        print(f"✅ Quantitative Analyst Success!")
        print(f"   Data points: {quant_result.metadata.get('data_points', 0)}")
        print(f"   Current price: ${quant_result.metadata.get('current_price', 0)}")
        print(f"   Confidence: {quant_result.confidence_score:.2f}")
        print(f"   Analysis preview: {quant_result.analysis[:200]}...")

if __name__ == "__main__":
    test_agents()
