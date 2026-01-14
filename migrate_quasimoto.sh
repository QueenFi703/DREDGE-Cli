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
