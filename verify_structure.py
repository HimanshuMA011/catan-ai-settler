#!/usr/bin/env python3
"""
Directory structure verification script for Catan Assistant.
Run this to ensure all files are in the correct locations.
"""

import os
from pathlib import Path
import sys

def verify_structure():
    """Verify the complete project structure."""
    
    required_files = [
        # Core application files
        "catan_assistant/__init__.py",
        "catan_assistant/core/__init__.py", 
        "catan_assistant/core/game_state.py",
        "catan_assistant/perception/__init__.py",
        "catan_assistant/perception/screen_capture.py",
        "catan_assistant/perception/state_extraction.py",
        "catan_assistant/reasoning/__init__.py",
        "catan_assistant/reasoning/dspy_module.py",
        "catan_assistant/orchestration/__init__.py",
        "catan_assistant/orchestration/langgraph_pipeline.py",
        "catan_assistant/orchestration/langchain_tools.py",
        "catan_assistant/data/__init__.py",
        "catan_assistant/data/training_generator.py",
        "catan_assistant/api/__init__.py",
        "catan_assistant/api/fastapi_app.py",
        "catan_assistant/utils/__init__.py",
        "catan_assistant/utils/config.py",
        "catan_assistant/utils/logger.py",
        
        # Configuration files
        "setup.py",
        "config.yaml", 
        ".env.example",
        
        # Docker files
        "docker/Dockerfile",
        "docker/docker-compose.yml", 
        "docker/requirements.txt",
    ]
    
    required_dirs = [
        "catan_assistant",
        "catan_assistant/core",
        "catan_assistant/perception", 
        "catan_assistant/reasoning",
        "catan_assistant/orchestration",
        "catan_assistant/data",
        "catan_assistant/api",
        "catan_assistant/utils",
        "docker",
    ]
    
    print("🔍 Verifying Catan Assistant project structure...")
    print("=" * 60)
    
    # Check directories
    missing_dirs = []
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Missing directory: {dir_path}")
            missing_dirs.append(dir_path)
    
    print()
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if Path(file_path).is_file():
            size = Path(file_path).stat().st_size
            print(f"✅ File: {file_path} ({size} bytes)")
        else:
            print(f"❌ Missing file: {file_path}")
            missing_files.append(file_path)
    
    print("=" * 60)
    
    # Summary
    if missing_dirs or missing_files:
        print(f"❌ Structure verification FAILED!")
        print(f"   Missing directories: {len(missing_dirs)}")
        print(f"   Missing files: {len(missing_files)}")
        print()
        print("Please ensure all files and directories are in the correct locations.")
        return False
    else:
        print("🎉 Structure verification PASSED!")
        print("All required files and directories are in place.")
        print()
        print("Next steps:")
        print("1. Set up your virtual environment: python -m venv venv")
        print("2. Activate it: source venv/bin/activate (or venv\\Scripts\\activate on Windows)")
        print("3. Install dependencies: pip install -r docker/requirements.txt")
        print("4. Install the package: pip install -e .")
        print("5. Configure your environment: cp .env.example .env (and add your OpenAI API key)")
        print("6. Run the application: uvicorn catan_assistant.api.fastapi_app:app --reload")
        print("7. Visit http://localhost:8000/docs to see the API documentation")
        return True

def create_missing_dirs():
    """Create any missing directories."""
    dirs_to_create = [
        "logs",
        "training_data", 
        "models",
        "tests",
        "notebooks",
        "training"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

if __name__ == "__main__":
    create_missing_dirs()
    success = verify_structure()
    sys.exit(0 if success else 1)