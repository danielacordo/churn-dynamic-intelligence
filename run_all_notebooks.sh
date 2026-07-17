set -e

TIMEOUT=300
FAILED=()

echo ""
echo "Churn as a Dynamic System - Notebook Execution "
echo ""

if [ ! -f "data/telco_churn.csv" ]; then
    echo "X data/telco_churn.csv not found."
    echo " Run: python download_data.py"
    echo " Or: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
    exit 1
fi

mkdir -p figures

NOTEBOOKS=(
    "01_eda.ipynb"
    "02_physical_variables.ipynb"
    "03_dynamic_model.ipynb"
    "04_phase_transitions.ipynb"
    "05_relaxation_time.ipynb"
    "06_error_propagation.ipynb"
    "07_final_model.ipynb"
    "08_survival_analysis.ipynb"
    "09_baseline_comparison.ipynb"
    "10_business_decisions.ipynb"
    "11_impact_metrics.ipynb"
    "sql_analysis_executed.ipynb")

TOTAL=${#NOTEBOOKS[@]}
PASSED=0
START=$(date +%s)

for nb in "${NOTEBOOKS[@]}"; do
    echo -n "  → $nb ... "
    if jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=$TIMEOUT \
        "notebooks/$nb" > /dev/null 2>&1; then
        echo "OK"
        PASSED=$((PASSED + 1))
    else
        echo "X"
        FAILED+=("$nb")
    fi
done

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo ""
echo " Results : $PASSED/$TOTAL notebooks executed"
echo " Time : ${ELAPSED}s"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo " Failed:"
    for nb in "${FAILED[@]}"; do echo "    ✗ $nb"; done
    echo ""
    echo " Re-run single notebook:"
    echo " jupyter nbconvert --to notebook --execute --inplace \\"
    echo "    --ExecutePreprocessor.timeout=300 notebooks/<name>.ipynb"
    exit 1
fi

echo ""
echo "  All notebooks completed."
echo "  Figures -> plots/"
echo "  Data -> data/telco_final.csv"
echo ""
echo "  Next:"
echo "    python main.py --simulate"
echo "    streamlit run app.py"
echo ""
echo ""
