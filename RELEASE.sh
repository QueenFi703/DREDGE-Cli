#!/bin/bash
# DREDGE Studio - PyPI Release Guide
# This script prepares and publishes DREDGE Studio to PyPI

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║           DREDGE STUDIO - PyPI RELEASE GUIDE                  ║"
echo "║              Step-by-step publication instructions            ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# Configuration
PACKAGE_NAME="dredge-studio"
VERSION="2.0.0"
REPO_URL="https://github.com/docker/dredge-cli-repo.git"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Verify Repository
echo -e "${BLUE}Step 1: Verifying repository...${NC}"
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository"
    exit 1
fi
echo -e "${GREEN}✓ Git repository verified${NC}"

# Step 2: Check Python
echo -e "${BLUE}Step 2: Checking Python version...${NC}"
python_version=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python ${python_version} found${NC}"

# Step 3: Install build dependencies
echo -e "${BLUE}Step 3: Installing build dependencies...${NC}"
pip install --upgrade setuptools wheel twine build
echo -e "${GREEN}✓ Build dependencies installed${NC}"

# Step 4: Clean previous builds
echo -e "${BLUE}Step 4: Cleaning previous builds...${NC}"
rm -rf build/ dist/ *.egg-info
echo -e "${GREEN}✓ Clean complete${NC}"

# Step 5: Build distribution
echo -e "${BLUE}Step 5: Building distribution...${NC}"
python -m build
echo -e "${GREEN}✓ Distribution built${NC}"

# Step 6: Check distribution
echo -e "${BLUE}Step 6: Checking distribution...${NC}"
twine check dist/*
echo -e "${GREEN}✓ Distribution verified${NC}"

# Step 7: Display artifacts
echo -e "${BLUE}Step 7: Distribution artifacts:${NC}"
ls -lh dist/
echo -e "${GREEN}✓ Ready for upload${NC}"

# Step 8: Instructions for upload
echo ""
echo -e "${YELLOW}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  NEXT STEPS - UPLOAD TO PyPI                                  ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Option A: Upload to TestPyPI (recommended for testing)"
echo "  twine upload --repository testpypi dist/*"
echo ""
echo "Option B: Upload to Production PyPI (requires credentials)"
echo "  twine upload dist/*"
echo ""
echo "Option C: Upload with token"
echo "  twine upload -u __token__ -p your_token_here dist/*"
echo ""
echo "Environment variable for token:"
echo "  export TWINE_PASSWORD='your_token_here'"
echo "  twine upload dist/*"
echo ""
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "1. Create PyPI account at https://pypi.org/account/register/"
echo "2. Generate API token at https://pypi.org/manage/account/tokens/"
echo "3. Create ~/.pypirc file with credentials"
echo "4. Or use environment variable: TWINE_PASSWORD"
echo ""
echo "~/.pypirc template:"
echo "  [testpypi]"
echo "  repository = https://test.pypi.org/legacy/"
echo "  username = __token__"
echo "  password = pypi-AgEIcHlwaS5vcmc..."
echo ""
echo "  [pypi]"
echo "  repository = https://upload.pypi.org/legacy/"
echo "  username = __token__"
echo "  password = pypi-AgEIcHlwaS5vcmc..."
echo ""
echo -e "${GREEN}Release build complete!${NC}"
echo ""
echo "Package: ${PACKAGE_NAME}"
echo "Version: ${VERSION}"
echo "Repository: ${REPO_URL}"
echo ""
echo "Files ready in ./dist/"
