#!/bin/bash
# Compile paper with BibTeX
# Usage: ./compile.sh

set -e  # Exit on error

check_pdflatex_error() {
    # Check if there are actual errors (lines starting with !) in the log
    if grep -q "^!" catastrophe_mitigation.log 2>/dev/null; then
        return 1
    fi
    # Check if PDF was created
    if [ ! -f catastrophe_mitigation.pdf ]; then
        return 1
    fi
    return 0
}

echo "Compiling paper..."
echo "Step 1/4: First pdflatex pass..."
pdflatex -interaction=nonstopmode catastrophe_mitigation.tex > /dev/null 2>&1 || true
if ! check_pdflatex_error; then
    echo "❌ Error in first pdflatex pass. Check catastrophe_mitigation.log for details."
    exit 1
fi
echo "  ✓ First pass complete"

echo "Step 2/4: Running BibTeX..."
bibtex catastrophe_mitigation > /dev/null 2>&1 || true
if [ ! -f catastrophe_mitigation.bbl ]; then
    echo "❌ Error in BibTeX. Check catastrophe_mitigation.blg for details."
    exit 1
fi
echo "  ✓ BibTeX complete"

echo "Step 3/4: Second pdflatex pass (including bibliography)..."
pdflatex -interaction=nonstopmode catastrophe_mitigation.tex > /dev/null 2>&1 || true
if ! check_pdflatex_error; then
    echo "❌ Error in second pdflatex pass. Check catastrophe_mitigation.log for details."
    exit 1
fi
echo "  ✓ Second pass complete"

echo "Step 4/4: Final pdflatex pass (resolving cross-references)..."
pdflatex -interaction=nonstopmode catastrophe_mitigation.tex > /dev/null 2>&1 || true
if ! check_pdflatex_error; then
    echo "❌ Error in final pdflatex pass. Check catastrophe_mitigation.log for details."
    exit 1
fi
echo "  ✓ Final pass complete"

echo "✓ Compilation complete! PDF: catastrophe_mitigation.pdf"


