#!/usr/bin/env python3
"""
IndoT5 Hybrid Paraphraser Web Application
Startup script for the web interface
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'flask',
        'torch',
        'transformers',
        'sentence_transformers',
        'numpy',
        'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Beberapa dependency belum terinstall:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Silakan install dependency terlebih dahulu:")
        print("   pip install -r requirements-neural.txt")
        print("\n   Atau install Flask saja jika yang lain sudah ada:")
        print("   pip install flask")
        return False
    
    return True

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'data/sinonim_extended.json',
        'data/transformation_rules.json',
        'data/stopwords_id.txt'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Beberapa file data tidak ditemukan:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n💡 Pastikan file data tersedia sebelum menjalankan aplikasi")
        return False
    
    return True

def main():
    """Main function to start the web application"""
    print("🚀 IndoT5 Hybrid Paraphraser Web Application")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists('app.py'):
        print("❌ File app.py tidak ditemukan!")
        print("💡 Pastikan Anda menjalankan script ini dari direktori project")
        sys.exit(1)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Check data files
    print("📄 Checking data files...")
    if not check_data_files():
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("\n🌐 Starting web server...")
    print("📍 Server akan berjalan di: http://localhost:5000")
    print("⏹️  Tekan Ctrl+C untuk menghentikan server")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app, initialize_paraphraser
        
        print("🔧 Initializing IndoT5 Hybrid Paraphraser...")
        initialize_paraphraser()
        
        print("✅ Paraphraser initialized successfully!")
        print("🌐 Web server is starting...")
        print("")
        print("📱 Buka browser dan akses: http://localhost:5000")
        print("🎯 Atau klik link di atas untuk mengakses interface")
        print("")
        
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server dihentikan oleh user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error menjalankan server: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Pastikan semua dependency terinstall")
        print("   2. Pastikan file data tersedia")
        print("   3. Periksa apakah port 5000 sedang digunakan")
        sys.exit(1)

if __name__ == '__main__':
    main() 
