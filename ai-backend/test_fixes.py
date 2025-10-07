#!/usr/bin/env python3
"""
Test script to validate the fixed app.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_imports():
    """Test that all imports work"""
    try:
        import app
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during import: {e}")
        return False

def test_dashboard_generator():
    """Test DashboardGenerator class"""
    try:
        from app import DashboardGenerator
        generator = DashboardGenerator()
        
        # Test data processing
        sample_data_context = {
            'db_path': 'test.db',
            'table_name': 'test_table',
            'columns': ['id', 'name', 'value'],
            'sample_data': [{'id': 1, 'name': 'test', 'value': 100}],
            'total_rows': 1
        }
        
        # Test fallback dashboard generation
        dashboard_code = generator.get_fallback_dashboard(sample_data_context, 'analytics')
        
        if len(dashboard_code) > 100 and 'streamlit' in dashboard_code:
            print("✅ Dashboard generation working")
            return True
        else:
            print("❌ Dashboard generation failed")
            return False
            
    except Exception as e:
        print(f"❌ DashboardGenerator test failed: {e}")
        return False

def test_helper_functions():
    """Test utility functions"""
    try:
        # Test that we can access the Flask app and its functions
        import app
        
        # Test that the Flask app is properly configured
        if hasattr(app, 'app') and hasattr(app.app, 'config'):
            print("✅ Flask app configuration working")
            return True
        else:
            print("❌ Flask app configuration failed")
            return False
            
    except Exception as e:
        print(f"❌ Helper functions test failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        from config import Config
        
        # Test basic config access
        if hasattr(Config, 'OLLAMA_URL') and hasattr(Config, 'AI_BACKEND_PORT'):
            print("✅ Configuration loading working")
            return True
        else:
            print("❌ Configuration loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    print("🧪 Testing fixed app.py...")
    print("-" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dashboard Generator", test_dashboard_generator),
        ("Helper Functions", test_helper_functions),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"   Test {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app should be working now.")
        print("\n📋 Next steps:")
        print("   1. Install missing dependencies: pip install streamlit plotly")
        print("   2. Start Ollama: ollama serve")
        print("   3. Run the app: python app.py")
        print("   4. Test the API endpoints")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)