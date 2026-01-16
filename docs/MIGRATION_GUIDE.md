# Quasimoto Migration Guide

## Transfer All Quasimoto Work to `QueenFi703/quasimoto-physics`

This guide provides step-by-step instructions to copy all Quasimoto-related files from the DREDGE repository to the quasimoto-physics repository.

---

## Quick Transfer Script

Save this as `migrate_quasimoto.sh` and run it:

```bash
#!/bin/bash

# Navigate to parent directory
cd ~

# Clone both repositories (if not already cloned)
if [ ! -d "DREDGE" ]; then
    git clone https://github.com/QueenFi703/DREDGE.git
fi

if [ ! -d "quasimoto-physics" ]; then
    git clone https://github.com/QueenFi703/quasimoto-physics.git
fi

# Copy all Quasimoto files
cp DREDGE/quasimoto_benchmark.py quasimoto-physics/
cp DREDGE/quasimoto_extended_benchmark.py quasimoto-physics/
cp DREDGE/quasimoto_6d_benchmark.py quasimoto-physics/
cp DREDGE/quasimoto_interference_benchmark.py quasimoto-physics/
cp DREDGE/quasimoto_paper.tex quasimoto-physics/

# Copy all PNG visualizations
cp DREDGE/quasimoto_comparison.png quasimoto-physics/
cp DREDGE/quasimoto_convergence.png quasimoto-physics/
cp DREDGE/quasimoto_4d_convergence.png quasimoto-physics/
cp DREDGE/quasimoto_6d_convergence.png quasimoto-physics/
cp DREDGE/quasimoto_6d_projection.png quasimoto-physics/
cp DREDGE/quasimoto_interference_comparison.png quasimoto-physics/
cp DREDGE/quasimoto_interference_convergence.png quasimoto-physics/

# Copy documentation
cp DREDGE/BENCHMARK_USAGE.md quasimoto-physics/
cp DREDGE/EXTENDED_BENCHMARK_README.md quasimoto-physics/
cp DREDGE/QUASIMOTO_6D_README.md quasimoto-physics/
cp DREDGE/EXPERIMENTATION_GUIDE.md quasimoto-physics/
cp DREDGE/RESUME_CONTENT.md quasimoto-physics/
cp DREDGE/LATEX_README.md quasimoto-physics/

# Copy requirements.txt
cp DREDGE/requirements.txt quasimoto-physics/

# Navigate to quasimoto-physics and commit
cd quasimoto-physics
git add .
git commit -m "Add complete Quasimoto benchmark suite with 1D/4D/6D extensions, interference basis, visualizations, and LaTeX paper"
git push origin main

echo "✅ Migration complete! All Quasimoto files transferred to quasimoto-physics repository."
```

---

## Manual Step-by-Step Instructions

If you prefer to do it manually:

### Step 1: Clone Both Repositories

```bash
cd ~
git clone https://github.com/QueenFi703/DREDGE.git
git clone https://github.com/QueenFi703/quasimoto-physics.git
```

### Step 2: Copy Python Scripts (4 files)

```bash
cp DREDGE/quasimoto_benchmark.py quasimoto-physics/
cp DREDGE/quasimoto_extended_benchmark.py quasimoto-physics/
cp DREDGE/quasimoto_6d_benchmark.py quasimoto-physics/
cp DREDGE/quasimoto_interference_benchmark.py quasimoto-physics/
```

### Step 3: Copy Visualizations (7 PNG files)

```bash
cp DREDGE/quasimoto_comparison.png quasimoto-physics/
cp DREDGE/quasimoto_convergence.png quasimoto-physics/
cp DREDGE/quasimoto_4d_convergence.png quasimoto-physics/
cp DREDGE/quasimoto_6d_convergence.png quasimoto-physics/
cp DREDGE/quasimoto_6d_projection.png quasimoto-physics/
cp DREDGE/quasimoto_interference_comparison.png quasimoto-physics/
cp DREDGE/quasimoto_interference_convergence.png quasimoto-physics/
```

### Step 4: Copy Documentation (7 files)

```bash
cp DREDGE/BENCHMARK_USAGE.md quasimoto-physics/
cp DREDGE/EXTENDED_BENCHMARK_README.md quasimoto-physics/
cp DREDGE/QUASIMOTO_6D_README.md quasimoto-physics/
cp DREDGE/EXPERIMENTATION_GUIDE.md quasimoto-physics/
cp DREDGE/RESUME_CONTENT.md quasimoto-physics/
cp DREDGE/LATEX_README.md quasimoto-physics/
cp DREDGE/quasimoto_paper.tex quasimoto-physics/
```

### Step 5: Copy Dependencies

```bash
cp DREDGE/requirements.txt quasimoto-physics/
```

### Step 6: Commit and Push

```bash
cd quasimoto-physics
git add .
git commit -m "Add complete Quasimoto benchmark suite

- 1D, 4D, and 6D wave function architectures
- Complex-valued interference basis implementation
- RFF and SIREN comparison baselines
- 7 publication-quality visualizations
- LaTeX paper and comprehensive documentation
- Experimentation guides and resume content
"
git push origin main
```

---

## Files Transferred (19 total)

### Python Scripts (4)
1. ✅ `quasimoto_benchmark.py` - Original 1D benchmark
2. ✅ `quasimoto_extended_benchmark.py` - 4D/6D/RFF/Interference extensions
3. ✅ `quasimoto_6d_benchmark.py` - Standalone 6D benchmark
4. ✅ `quasimoto_interference_benchmark.py` - Complex-valued waves

### Visualizations (7)
5. ✅ `quasimoto_comparison.png` - 4-panel prediction comparison
6. ✅ `quasimoto_convergence.png` - Training curves
7. ✅ `quasimoto_4d_convergence.png` - 4D training
8. ✅ `quasimoto_6d_convergence.png` - 6D training
9. ✅ `quasimoto_6d_projection.png` - 6D 2D projection
10. ✅ `quasimoto_interference_comparison.png` - Interference predictions
11. ✅ `quasimoto_interference_convergence.png` - Interference training

### Documentation (7)
12. ✅ `BENCHMARK_USAGE.md` - Architecture guide
13. ✅ `EXTENDED_BENCHMARK_README.md` - Extended features docs
14. ✅ `QUASIMOTO_6D_README.md` - 6D documentation
15. ✅ `EXPERIMENTATION_GUIDE.md` - 8 advanced experiments
16. ✅ `RESUME_CONTENT.md` - Professional resume content
17. ✅ `LATEX_README.md` - LaTeX compilation guide
18. ✅ `quasimoto_paper.tex` - Publication-ready paper

### Dependencies (1)
19. ✅ `requirements.txt` - torch, numpy, matplotlib

---

## After Migration

Once files are in `quasimoto-physics`, you can:

1. **Run benchmarks:**
   ```bash
   cd quasimoto-physics
   pip install -r requirements.txt
   python quasimoto_benchmark.py
   python quasimoto_extended_benchmark.py
   python quasimoto_6d_benchmark.py
   python quasimoto_interference_benchmark.py
   ```

2. **Compile the paper:**
   ```bash
   pdflatex quasimoto_paper.tex
   pdflatex quasimoto_paper.tex  # Run twice for references
   ```

3. **Create a README.md** for the repository linking to all documentation

---

## Verification

After migration, verify all files are present:

```bash
cd quasimoto-physics
ls -la quasimoto*.py quasimoto*.png *.md quasimoto_paper.tex requirements.txt
```

You should see all 19 files listed.

---

## Need Help?

If you encounter issues:
- Ensure you have write access to `QueenFi703/quasimoto-physics`
- Check that Git is properly configured with your GitHub credentials
- Make sure you're on the correct branch (usually `main` or `master`)

---

**Created**: 2026-01-13  
**Author**: GitHub Copilot  
**Purpose**: Migrate Quasimoto benchmark suite from DREDGE to quasimoto-physics repository
