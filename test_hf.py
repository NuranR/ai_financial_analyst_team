"""Simple HuggingFace test script."""

import requests

def test_hf_api():
    """Test different HuggingFace API endpoints."""
    
    api_token = "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"
    headers = {"Authorization": f"Bearer {api_token}"}
    
    # Test different models and endpoints
    models_to_test = [
        "gpt2",
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill"
    ]
    
    for model in models_to_test:
        print(f"\n🧪 Testing {model}...")
        
        # Try text generation endpoint
        url = f"https://api-inference.huggingface.co/models/{model}"
        
        payload = {
            "inputs": "Apple Inc. is a technology company that",
            "parameters": {"max_new_tokens": 50}
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success: {result}")
                return model, url  # Return first working model
            else:
                print(f"   ❌ Error: {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    return None, None

if __name__ == "__main__":
    working_model, working_url = test_hf_api()
    if working_model:
        print(f"\n✅ Working model found: {working_model}")
    else:
        print("\n❌ No working models found")
