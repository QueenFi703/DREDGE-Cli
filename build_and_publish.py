#!/usr/bin/env python3
"""
DREDGE Studio v2.0 - Automated Build & Publish Script
Run this from anywhere - handles directory navigation automatically
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors"""
    if description:
        print(f"\n{'='*70}")
        print(f"STEP: {description}")
        print(f"{'='*70}")
        print(f"Command: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║            DREDGE STUDIO v2.0 - Build & Publish Script                    ║
║                  Automated PyPI Distribution                               ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Find dredge-cli-repo directory
    print("Step 1: Locating dredge-cli-repo...")
    current_dir = Path.cwd()
    
    # Check if we're already in dredge-cli-repo
    if (current_dir / "setup.py").exists():
        dredge_dir = current_dir
        print(f"✅ Found in current directory: {dredge_dir}")
    # Check if dredge-cli-repo exists in current directory
    elif (current_dir / "dredge-cli-repo" / "setup.py").exists():
        dredge_dir = current_dir / "dredge-cli-repo"
        print(f"✅ Found dredge-cli-repo: {dredge_dir}")
    else:
        print("❌ Could not find dredge-cli-repo with setup.py")
        print(f"   Current directory: {current_dir}")
        print(f"   Looking for: {current_dir}/dredge-cli-repo/setup.py")
        sys.exit(1)
    
    # Change to the directory
    os.chdir(dredge_dir)
    print(f"📁 Working directory: {os.getcwd()}")
    
    # Check Python version
    print(f"\n✅ Python: {sys.version.split()[0]}")
    print(f"✅ Location: {sys.executable}")
    
    # Step 1: Upgrade pip
    if not run_command("python -m pip install --upgrade pip", "Upgrade pip"):
        print("⚠️  Warning: pip upgrade had issues, continuing...")
    
    # Step 2: Install build tools
    print("\n" + "="*70)
    print("STEP: Install build tools (build, twine, setuptools, wheel)")
    print("="*70)
    cmd = "python -m pip install build twine setuptools wheel"
    print(f"Command: {cmd}\n")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print("✅ Build tools installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install build tools")
        sys.exit(1)
    
    # Step 3: Clean old builds
    print("\n" + "="*70)
    print("STEP: Clean old builds")
    print("="*70)
    for pattern in ["build", "dist", "*.egg-info", "src/*.egg-info"]:
        import glob
        for path in glob.glob(pattern):
            try:
                import shutil
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"✅ Removed directory: {path}")
                else:
                    os.remove(path)
                    print(f"✅ Removed file: {path}")
            except Exception as e:
                print(f"⚠️  Could not remove {path}: {e}")
    
    # Step 4: Build distribution
    if not run_command("python -m build", "Build distribution packages"):
        print("❌ Build failed")
        sys.exit(1)
    
    # Step 5: Verify
    if not run_command("twine check dist/*", "Verify distribution integrity"):
        print("❌ Verification failed")
        sys.exit(1)
    
    # Step 6: Show what was built
    print("\n" + "="*70)
    print("BUILT PACKAGES:")
    print("="*70)
    dist_dir = Path("dist")
    if dist_dir.exists():
        for file in sorted(dist_dir.glob("*")):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"✅ {file.name} ({size_mb:.2f} MB)")
    
    # Final instructions
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("""
1. CREATE PyPI ACCOUNT (if you don't have one):
   Go to: https://pypi.org/account/register/
   
2. GENERATE API TOKEN:
   Go to: https://pypi.org/manage/account/tokens/
   Create token with "Entire account" scope
   
3. CONFIGURE CREDENTIALS:
   Create ~/.pypirc file or set TWINE_PASSWORD env var
   
4. UPLOAD TO PRODUCTION:
   twine upload dist/*
   
5. VERIFY INSTALLATION:
   pip install dredge-studio
   
Optional: Test on TestPyPI first
   twine upload --repository testpypi dist/*
    """)
    
    # Ask if user wants to upload now
    print("\n" + "="*70)
    response = input("Ready to upload to PyPI now? (yes/no): ").strip().lower()
    
    if response == 'yes':
        print("\nUploading to PyPI...")
        if run_command("twine upload dist/*", "Upload to PyPI"):
            print("\n✅ Successfully uploaded to PyPI!")
            print("View at: https://pypi.org/project/dredge-studio/")
        else:
            print("\n❌ Upload failed - check credentials and try again")
            print("Command: twine upload dist/*")
    else:
        print("\n📦 Packages ready in dist/ directory")
        print("When ready, run: twine upload dist/*")
    
    print("\n✅ Done!")

if __name__ == "__main__":
    main()
