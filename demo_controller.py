#!/usr/bin/env python3
"""
A-FIN AI Provider Controller
Switch between Gemini, Hugging Face, and Demo modes.
"""

import os
import sys
from pathlib import Path

def set_ai_mode(mode: str = "gemini"):
    """Set AI provider mode: gemini, huggingface, or demo."""
    
    env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        print("❌ .env file not found!")
        return False
    
    # Read current .env
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update settings
    updated_lines = []
    settings_found = {"demo_mode": False, "use_huggingface": False}
    
    for line in lines:
        if line.startswith('DEMO_MODE='):
            updated_lines.append(f'DEMO_MODE={"true" if mode == "demo" else "false"}\n')
            settings_found["demo_mode"] = True
        elif line.startswith('USE_HUGGINGFACE='):
            updated_lines.append(f'USE_HUGGINGFACE={"true" if mode == "huggingface" else "false"}\n')
            settings_found["use_huggingface"] = True
        else:
            updated_lines.append(line)
    
    # Add missing settings
    if not settings_found["demo_mode"]:
        updated_lines.append(f'DEMO_MODE={"true" if mode == "demo" else "false"}\n')
    if not settings_found["use_huggingface"]:
        updated_lines.append(f'USE_HUGGINGFACE={"true" if mode == "huggingface" else "false"}\n')
    
    # Write back to .env
    with open(env_path, 'w') as f:
        f.writelines(updated_lines)
    
    if mode == "demo":
        print("🎭 Switched to DEMO MODE")
        print("   - Uses pre-computed responses")
        print("   - No API calls required")
        print("   - Perfect for demonstrations")
    elif mode == "huggingface":
        print("🤗 Switched to HUGGING FACE MODE")
        print("   - Uses HuggingFace Inference API")
        print("   - Requires HUGGINGFACE_API_TOKEN")
        print("   - Free alternative to Gemini")
    else:  # gemini
        print("🌟 Switched to GEMINI MODE")
        print("   - Uses Google Gemini API")
        print("   - Requires GEMINI_API_KEY")
        print("   - Best quality responses")
    
    return True

def show_status():
    """Show current AI provider status."""
    env_path = Path(__file__).parent / ".env"
    
    if not env_path.exists():
        print("❌ .env file not found!")
        return
    
    with open(env_path, 'r') as f:
        content = f.read()
    
    # Check current mode
    demo_mode = 'DEMO_MODE=true' in content
    hf_mode = 'USE_HUGGINGFACE=true' in content
    
    if demo_mode:
        print("🎭 Currently in DEMO MODE")
        print("   - No API calls required")
        print("   - Uses pre-computed responses")
    elif hf_mode:
        print("🤗 Currently in HUGGING FACE MODE")
        print("   - Uses HuggingFace Inference API")
        print("   - Free tier available")
    else:
        print("� Currently in GEMINI MODE")
        print("   - Uses Google Gemini API")
        print("   - Requires API quotas")

def setup_huggingface():
    """Guide user through HuggingFace setup."""
    print("\n🤗 HUGGING FACE SETUP GUIDE")
    print("=" * 40)
    print("1. Go to: https://huggingface.co/join")
    print("2. Create a free account")
    print("3. Go to: https://huggingface.co/settings/tokens")
    print("4. Click 'New token'")
    print("5. Name: 'A-FIN Financial Analysis'")
    print("6. Type: 'Read'")
    print("7. Copy the token (starts with 'hf_')")
    print("8. Add to .env file: HUGGINGFACE_API_TOKEN=hf_your_token_here")
    print("\n💡 TIP: HuggingFace is completely FREE with good quality!")

if __name__ == "__main__":
    print("🏦 A-FIN AI Provider Controller")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        show_status()
        print("\nUsage:")
        print("  python demo_controller.py gemini     # Use Gemini AI (best quality)")
        print("  python demo_controller.py huggingface # Use HuggingFace (free)")
        print("  python demo_controller.py demo       # Use demo mode (offline)")
        print("  python demo_controller.py status     # Show current mode")
        print("  python demo_controller.py setup-hf   # HuggingFace setup guide")
    
    elif sys.argv[1].lower() in ['gemini', 'google']:
        set_ai_mode('gemini')
        
    elif sys.argv[1].lower() in ['huggingface', 'hf', 'hugging-face']:
        set_ai_mode('huggingface')
        
    elif sys.argv[1].lower() in ['demo', 'offline']:
        set_ai_mode('demo')
        
    elif sys.argv[1].lower() in ['status', 'show']:
        show_status()
        
    elif sys.argv[1].lower() in ['setup-hf', 'setup-huggingface']:
        setup_huggingface()
        
    else:
        print("❌ Invalid option. Use 'gemini', 'huggingface', 'demo', or 'status'")
