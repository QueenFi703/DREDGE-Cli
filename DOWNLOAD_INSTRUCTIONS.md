# µH-iOS Whitepaper - Download Instructions

## Available Archive Formats

Two archive formats have been created for your convenience:

### 1. TAR.GZ Archive (Recommended for Linux/Mac)
- **File**: `uhios-whitepaper-package.tar.gz`
- **Size**: 15 KB
- **Format**: Compressed tar archive

**Download and Extract**:
```bash
# Extract the archive
tar -xzf uhios-whitepaper-package.tar.gz

# Navigate to the directory
cd uhios-whitepaper-package

# Compile the whitepaper
make
```

### 2. ZIP Archive (Recommended for Windows)
- **File**: `uhios-whitepaper-package.zip`
- **Size**: 18 KB
- **Format**: Standard ZIP archive

**Download and Extract**:
```bash
# Extract the archive (Linux/Mac)
unzip uhios-whitepaper-package.zip

# Or use your system's built-in extraction tool (Windows/Mac)

# Navigate to the directory
cd uhios-whitepaper-package

# Compile the whitepaper
make
```

## Package Contents

Both archives contain the same files:

```
uhios-whitepaper-package/
├── README.md                   # Package overview and quick start
├── uhios-whitepaper.tex        # Main LaTeX source (534 lines)
├── Makefile                    # Build system
├── README-whitepaper.md        # Compilation guide
├── SUBMISSION_GUIDE.md         # Academic submission strategy
└── LICENSE                     # MIT License
```

## Files in Repository

The archives are located in the repository root:
- `/home/runner/work/DREDGE/DREDGE/uhios-whitepaper-package.tar.gz`
- `/home/runner/work/DREDGE/DREDGE/uhios-whitepaper-package.zip`

## Accessing from Git Repository

You can also clone the repository and access the files directly:

```bash
# Clone the repository
git clone https://github.com/QueenFi703/DREDGE.git

# Navigate to the docs directory
cd DREDGE/docs

# Compile the whitepaper
make
```

## What's Inside

### uhios-whitepaper.tex (Main Document)
- Professional LaTeX formatting
- Abstract and introduction
- Formal threat model with TikZ diagram
- Mathematical system model (Σ, VMState, capabilities)
- 4 formal invariants as theorems
- Implementation details (Rust/Swift/C)
- Comprehensive evaluation (TCB, verification, performance)
- Related work and positioning
- Limitations and future work
- Appendices with proofs and specifications

### Makefile (Build System)
Simple commands:
- `make` - Compile PDF
- `make clean` - Remove build artifacts
- `make view` - Open PDF in viewer
- `make help` - Show help

### README-whitepaper.md (Compilation Guide)
- LaTeX prerequisites
- Installation instructions
- Submission checklist
- Citation format

### SUBMISSION_GUIDE.md (Academic Strategy)
- Target venues (SOSP, OSDI, EuroSys, IEEE S&P, USENIX Security)
- Submission preparation steps
- Response strategies for reviewers
- Post-acceptance guidance
- Timeline examples

## Quick Compilation Test

After extracting either archive:

```bash
cd uhios-whitepaper-package

# Check if LaTeX is installed
pdflatex --version

# Compile (runs twice for references)
make

# View the generated PDF
ls -lh uhios-whitepaper.pdf
```

Expected output: `uhios-whitepaper.pdf` (~200-300 KB)

## Prerequisites

To compile the whitepaper, you need:

**LaTeX Distribution:**
- Linux: `sudo apt-get install texlive-full`
- macOS: Install [MacTeX](https://www.tug.org/mactex/)
- Windows: Install [MiKTeX](https://miktex.org/)

**Required Packages** (usually included):
- amsmath, amssymb, amsthm
- tikz
- listings
- hyperref
- authblk

## Troubleshooting

**"pdflatex: command not found"**
- Install a LaTeX distribution (see Prerequisites)

**Missing packages**
- Use your LaTeX package manager to install missing dependencies
- On TeX Live: `tlmgr install <package-name>`
- On MiKTeX: Packages install automatically on first use

**Compilation errors**
- Ensure you're using a recent LaTeX distribution
- Try `make clean` then `make` again

## Support

For questions or issues:
- GitHub Issues: https://github.com/QueenFi703/DREDGE/issues
- Repository: https://github.com/QueenFi703/DREDGE

## Next Steps

1. Extract the archive
2. Read `README.md` in the package
3. Compile with `make`
4. Review the generated PDF
5. Consult `SUBMISSION_GUIDE.md` for publication venues

---

**Created**: January 2026
**Archive Version**: 1.0
**Repository**: https://github.com/QueenFi703/DREDGE
