"""Simple test script to debug NewsAPI"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.news_api import NewsAPI
import json

def test_news_api():
    """Test NewsAPI with detailed logging"""
    print("🔍 Testing NewsAPI...")
    
    api = NewsAPI()
    print(f"✅ API Key configured: {bool(api.api_key)}")
    print(f"✅ Base URL: {api.base_url}")
    
    # Test with Apple
    print("\n📰 Fetching news for Apple (AAPL)...")
    articles = api.get_company_news("Apple Inc.", "AAPL", days_back=7, max_articles=10)
    
    print(f"\n📊 Results:")
    print(f"Total articles fetched: {len(articles)}")
    
    if articles:
        print(f"\n📰 Sample articles:")
        for i, article in enumerate(articles[:3]):
            print(f"\n--- Article {i+1} ---")
            print(f"Title: {article['title']}")
            print(f"Source: {article['source']}")
            print(f"Published: {article['published_at']}")
            print(f"Relevance Score: {article['relevance_score']}")
            print(f"Description: {(article['description'] or 'No description')[:200]}...")
    else:
        print("❌ No articles found!")
        
    # Test direct API call
    print(f"\n🔍 Testing direct API call...")
    import requests
    
    params = {
        'q': 'Apple OR AAPL AND (stock OR shares OR earnings)',
        'sortBy': 'relevancy',
        'pageSize': 5,
        'language': 'en',
        'apiKey': api.api_key
    }
    
    try:
        response = requests.get(f"{api.base_url}/everything", params=params)
        print(f"Direct API status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Total results from API: {data.get('totalResults', 0)}")
            print(f"Articles in response: {len(data.get('articles', []))}")
            
            if data.get('articles'):
                for i, article in enumerate(data['articles'][:2]):
                    print(f"Raw article {i+1}: {article.get('title', 'No title')}")
        else:
            print(f"API Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Direct API test failed: {e}")

if __name__ == "__main__":
    test_news_api()
