#!/usr/bin/env python3
"""
Simple PyPI Build & Upload for DREDGE Studio
Handles all the setup automatically
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_cmd(cmd, desc=""):
    """Run command with error handling"""
    if desc:
        print(f"\n{'='*70}")
        print(f"► {desc}")
        print(f"{'='*70}")
    print(f"$ {cmd}\n")
    result = os.system(cmd)
    if result != 0:
        print(f"\n❌ Command failed with code {result}")
        return False
    return True

def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         DREDGE STUDIO - Simple PyPI Build & Upload                        ║
║                    Fixed for missing files                                 ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Find dredge-cli-repo
    if not Path("setup.py").exists() and Path("dredge-cli-repo/setup.py").exists():
        os.chdir("dredge-cli-repo")
        print("✅ Changed to dredge-cli-repo directory\n")
    
    if not Path("setup.py").exists():
        print("❌ ERROR: setup.py not found!")
        print("   Make sure you're in dredge-cli-repo directory")
        sys.exit(1)
    
    print(f"📁 Working in: {os.getcwd()}\n")
    
    # Step 1: Clean
    print("Step 1: Cleaning old builds...")
    for d in ["build", "dist"]:
        if Path(d).exists():
            shutil.rmtree(d)
            print(f"  ✓ Removed {d}/")
    
    for f in Path(".").glob("*.egg-info"):
        shutil.rmtree(f)
        print(f"  ✓ Removed {f}/")
    
    # Step 2: Install tools
    print("\nStep 2: Installing build tools...")
    run_cmd("python -m pip install --upgrade pip setuptools wheel build twine", 
            "Installing dependencies")
    
    # Step 3: Build
    print("\nStep 3: Building package...")
    if not run_cmd("python -m build", "Building distribution"):
        print("\n❌ Build failed - checking dist directory...")
        sys.exit(1)
    
    # Step 4: Show what was built
    print("\n" + "="*70)
    print("✅ BUILD COMPLETE - Files created:")
    print("="*70)
    dist_files = list(Path("dist").glob("*"))
    if not dist_files:
        print("❌ ERROR: No files in dist/ directory!")
        sys.exit(1)
    
    for f in sorted(dist_files):
        size = f.stat().st_size / (1024 * 1024)
        print(f"  • {f.name} ({size:.2f} MB)")
    
    # Step 5: Check (optional)
    print("\nStep 4: Verifying package...")
    try:
        result = subprocess.run(
            "twine check dist/*",
            shell=True,
            capture_output=True,
            text=True
        )
        if "Passed" in result.stdout or "passed" in result.stdout:
            print("  ✅ Package verification passed")
        else:
            print("  ⚠️  Package check output:")
            print(result.stdout)
            if result.returncode != 0:
                print(result.stderr)
    except Exception as e:
        print(f"  ⚠️  Could not verify: {e}")
    
    # Final instructions
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                         NEXT STEPS                                        ║
╚════════════════════════════════════════════════════════════════════════════╝

✅ Package built successfully in dist/

To upload to PyPI:

1. Create PyPI account (if needed):
   https://pypi.org/account/register/

2. Generate API token:
   https://pypi.org/manage/account/tokens/

3. Upload the package:
   twine upload dist/*

4. When prompted:
   Username: __token__
   Password: (paste your token here)

Or use environment variable:
   export TWINE_PASSWORD="pypi-AgE..."
   twine upload dist/*

✅ Done! The package is ready for distribution.
    """)

if __name__ == "__main__":
    main()
