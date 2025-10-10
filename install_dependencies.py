#!/usr/bin/env python3
"""
Installation script for DFS Optimizer enhanced features
"""
import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🏈 DFS Optimizer Enhanced Features Installer")
    print("=" * 50)
    
    # Read requirements
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found!")
        return
    
    with open(requirements_file, 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"📦 Installing {len(packages)} packages...")
    
    failed_packages = []
    
    for package in packages:
        print(f"Installing {package}...", end=' ')
        if install_package(package):
            print("✅")
        else:
            print("❌")
            failed_packages.append(package)
    
    print("\n" + "=" * 50)
    
    if failed_packages:
        print(f"⚠️ Failed to install: {', '.join(failed_packages)}")
        print("\n💡 Try installing manually:")
        for package in failed_packages:
            print(f"   pip install {package}")
    else:
        print("🎉 All packages installed successfully!")
        print("\n🚀 You can now run the enhanced DFS optimizer:")
        print("   streamlit run dfs_optimizer_app.py")

if __name__ == "__main__":
    main()