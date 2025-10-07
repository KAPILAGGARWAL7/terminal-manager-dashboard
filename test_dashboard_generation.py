#!/usr/bin/env python3
"""
Quick test of the dashboard generation functionality
"""

import sys
import os
import json

# Add the ai-backend directory to path
sys.path.append(r"c:\Users\kapil\Downloads\terminal_manager-18e1bd7106182a5f9d5f0048dcb08d4444584b2b\ai-backend")

def test_dashboard_generation():
    """Test dashboard generation without Flask"""
    try:
        from app import DashboardGenerator
        
        generator = DashboardGenerator()
        
        # Test data context
        test_data_context = {
            'db_path': r'c:\Users\kapil\Downloads\terminal_manager-18e1bd7106182a5f9d5f0048dcb08d4444584b2b\data\test.db',
            'table_name': 'sales_data',
            'columns': ['date', 'product', 'revenue', 'quantity'],
            'sample_data': [
                {'date': '2024-01-01', 'product': 'Widget A', 'revenue': 1000, 'quantity': 10},
                {'date': '2024-01-02', 'product': 'Widget B', 'revenue': 1500, 'quantity': 15}
            ],
            'total_rows': 2,
            'excel_source': 'sales_data.xlsx'
        }
        
        print("🧪 Testing dashboard code generation...")
        
        # Generate dashboard code
        dashboard_code = generator.generate_dashboard_code(
            "Create a sales analytics dashboard with revenue trends", 
            test_data_context
        )
        
        if dashboard_code and len(dashboard_code) > 500:
            print(f"✅ Dashboard code generated successfully ({len(dashboard_code)} chars)")
            
            # Try to create the file
            dashboard_id = "test_1234"
            dashboard_path = generator.create_dashboard_file(dashboard_code, dashboard_id)
            
            if dashboard_path and os.path.exists(dashboard_path):
                print(f"✅ Dashboard file created: {dashboard_path}")
                
                # Show preview
                print("\\n📄 Code preview (first 500 chars):")
                print("-" * 50)
                print(dashboard_code[:500] + "..." if len(dashboard_code) > 500 else dashboard_code)
                print("-" * 50)
                
                return True
            else:
                print("❌ Failed to create dashboard file")
                return False
        else:
            print(f"❌ Dashboard code generation failed (length: {len(dashboard_code) if dashboard_code else 0})")
            if dashboard_code:
                print("Generated code:", dashboard_code[:200])
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Dashboard Generation...")
    print("=" * 60)
    
    success = test_dashboard_generation()
    
    if success:
        print("\\n🎉 All tests passed! Dashboard generation is working.")
        print("\\n📋 To start the web server:")
        print("   cd ai-backend")
        print("   python app.py")
        print("\\n📋 To test via API:")
        print('   curl -X POST http://localhost:5247/api/dashboard/generate \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{"prompt": "Create a sales dashboard"}\'')
    else:
        print("\\n❌ Tests failed. Please check the errors above.")