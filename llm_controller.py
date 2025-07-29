#!/usr/bin/env python3
"""
A-FIN LLM Mode Controller
Switch between Gemini, HuggingFace, and Demo modes.
"""

import os
import sys
from pathlib import Path

def set_llm_mode(mode: str = "huggingface"):
    """Set the LLM mode: gemini, huggingface, or demo."""
    
    # Find .env file
    env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        print("❌ .env file not found!")
        return False
    
    # Read current .env
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update settings
    updated_lines = []
    demo_mode_found = False
    use_hf_found = False
    
    for line in lines:
        if line.startswith('DEMO_MODE='):
            updated_lines.append(f'DEMO_MODE={"true" if mode == "demo" else "false"}\n')
            demo_mode_found = True
        elif line.startswith('USE_HUGGINGFACE='):
            updated_lines.append(f'USE_HUGGINGFACE={"true" if mode == "huggingface" else "false"}\n')
            use_hf_found = True
        else:
            updated_lines.append(line)
    
    # Add missing settings
    if not demo_mode_found:
        updated_lines.append(f'DEMO_MODE={"true" if mode == "demo" else "false"}\n')
    if not use_hf_found:
        updated_lines.append(f'USE_HUGGINGFACE={"true" if mode == "huggingface" else "false"}\n')
    
    # Write back to .env
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    mode_icons = {
        "gemini": "🔮 GEMINI API MODE",
        "huggingface": "🤗 HUGGING FACE MODE", 
        "demo": "🎭 DEMO MODE"
    }
    
    print(f"✅ Switched to {mode_icons.get(mode, mode.upper())}")
    return True

def test_huggingface():
    """Test HuggingFace API connection."""
    try:
        from config.huggingface_client import test_huggingface_client
        from config.settings import settings
        
        if settings.huggingface_api_token:
            print("🧪 Testing HuggingFace connection...")
            test_huggingface_client(settings.huggingface_api_token)
        else:
            print("❌ HuggingFace API token not found in settings")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def show_status():
    """Show current LLM mode status."""
    env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        print("❌ .env file not found!")
        return
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    if 'DEMO_MODE=true' in content:
        print("🎭 Currently in DEMO MODE")
        print("   - Uses pre-computed responses")
        print("   - No API calls")
        print("   - Perfect for demonstrations")
    elif 'USE_HUGGINGFACE=true' in content:
        print("🤗 Currently in HUGGING FACE MODE")
        print("   - Uses Llama-3.1-8B-Instruct")
        print("   - Free Hugging Face API")
        print("   - High-quality analysis")
    else:
        print("🔮 Currently in GEMINI API MODE")
        print("   - Uses Google Gemini")
        print("   - Requires API quotas")
        print("   - Real-time analysis")

if __name__ == "__main__":
    print("🏦 A-FIN LLM Mode Controller")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        show_status()
        print("\nUsage:")
        print("  python llm_controller.py gemini      # Use Gemini API")
        print("  python llm_controller.py huggingface # Use HuggingFace API")
        print("  python llm_controller.py demo        # Use demo mode")
        print("  python llm_controller.py test        # Test HuggingFace")
        print("  python llm_controller.py status      # Show current status")
    
    elif sys.argv[1].lower() in ['gemini', 'google']:
        set_llm_mode('gemini')
        
    elif sys.argv[1].lower() in ['huggingface', 'hf', 'llama']:
        set_llm_mode('huggingface')
        
    elif sys.argv[1].lower() in ['demo', 'offline']:
        set_llm_mode('demo')
        
    elif sys.argv[1].lower() in ['test']:
        test_huggingface()
        
    elif sys.argv[1].lower() in ['status', 'show']:
        show_status()
        
    else:
        print("❌ Invalid option. Use 'gemini', 'huggingface', 'demo', 'test', or 'status'")
