#!/usr/bin/env bash
#
# tools/run_contract_suite.sh
# CI-friendly contract compliance test runner (Hardened v0.4.1)
#
# Usage: ./tools/run_contract_suite.sh [--allow-empty] [--allow-empty-events]
# Exits non-zero on any failure, including 0 tests or 0 events
#

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Parse optional flags
ALLOW_EMPTY=0
ALLOW_EMPTY_EVENTS=0
for arg in "$@"; do
    case "$arg" in
        --allow-empty) ALLOW_EMPTY=1 ;;
        --allow-empty-events) ALLOW_EMPTY_EVENTS=1 ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "CGF Contract Compliance Suite (Hardened)"
echo "========================================"
echo ""
echo "Flags: --allow-empty=${ALLOW_EMPTY}, --allow-empty-events=${ALLOW_EMPTY_EVENTS}"
echo ""

# Check Python availability
if ! command -v python3 > /dev/null 2>&1; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Python: ${PYTHON_VERSION}"
echo "Repo root: ${REPO_ROOT}"
echo ""

# Clean runtime directories before test
echo "[1/5] Cleaning runtime directories..."
rm -rf "${REPO_ROOT}/cgf_data" "${REPO_ROOT}/langgraph_cgf_data" "${REPO_ROOT}/openclaw_adapter_data"
rm -rf "${REPO_ROOT}/__pycache__" "${REPO_ROOT}/.pytest_cache" "${REPO_ROOT}/outputs"
echo -e "${GREEN}✓ Runtime directories cleaned${NC}"
echo ""

# Check if CGF is already running
echo "[2/5] Checking CGF server..."
CGF_PID=""
CGF_WAS_RUNNING=0

check_cgf_health() {
    curl -s http://127.0.0.1:8080/health > /dev/null 2>&1 || curl -s http://127.0.0.1:8080/docs > /dev/null 2>&1
}

if check_cgf_health; then
    echo -e "${GREEN}✓ CGF server already running (using existing)${NC}"
    CGF_WAS_RUNNING=1
    CGF_ALREADY_RUNNING=1
else
    CGF_ALREADY_RUNNING=0
fi

cleanup() {
    # Only kill if we started it
    if [ "${CGF_WAS_RUNNING}" -eq 0 ] && [ -n "$CGF_PID" ] && kill -0 "$CGF_PID" 2>/dev/null; then
        echo "Cleaning up CGF server (PID: $CGF_PID)..."
        kill "$CGF_PID" 2>/dev/null
        wait "$CGF_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start CGF server only if not already running
if [ "${CGF_ALREADY_RUNNING}" -eq 0 ]; then
    cd "${REPO_ROOT}"
    python3 server/cgf_server_v03.py &
    CGF_PID=$!
    echo "CGF server started (PID: $CGF_PID)"
fi

# Wait for health endpoint
echo "Waiting for CGF server to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0
CGF_READY=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://127.0.0.1:8080/health > /dev/null 2>&1; then
        CGF_READY=1
        break
    fi
    # Also check /docs as fallback
    if curl -s http://127.0.0.1:8080/docs > /dev/null 2>&1; then
        CGF_READY=1
        break
    fi
    sleep 0.5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $CGF_READY -eq 0 ]; then
    echo -e "${RED}ERROR: CGF server failed to start within $((MAX_RETRIES * 500))ms${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CGF server ready (http://127.0.0.1:8080)${NC}"
echo ""

# Run pytest
echo "[3/5] Running contract compliance tests..."
cd "${REPO_ROOT}"

# Capture output to file for parsing
PYTEST_OUTPUT=$(mktemp)
set +e
python3 -m pytest tools/contract_compliance_tests.py -v --tb=short > "$PYTEST_OUTPUT" 2>&1
EXIT_CODE=$?

# Parse BEFORE cat/rm
TESTS_COLLECTED=0
TESTS_PASSED=0
TESTS_FAILED=0

if grep -q "collected" "$PYTEST_OUTPUT" 2>/dev/null || true; then
    # Extract numbers from pytest summary line
    if grep -E "^tools/" "$PYTEST_OUTPUT" | grep -E "PASSED" > /dev/null 2>&1; then
        # Count PASSED lines in test output (lines starting with tools/ that contain PASSED)
        TESTS_PASSED=$(grep -E "^tools/.*PASSED" "$PYTEST_OUTPUT" | wc -l | tr -d ' ')
    fi
    if grep -E "^tools/" "$PYTEST_OUTPUT" | grep -E "FAILED" > /dev/null 2>&1; then
        TESTS_FAILED=$(grep -E "^tools/.*FAILED" "$PYTEST_OUTPUT" | wc -l | tr -d ' ')
    fi
    if grep -q "collected" "$PYTEST_OUTPUT" 2>/dev/null; then
        TESTS_COLLECTED=$(grep -oE "collected [0-9]+" "$PYTEST_OUTPUT" | head -1 | grep -oE "[0-9]+")
    fi
fi

# Default to 0 if empty
TESTS_COLLECTED=${TESTS_COLLECTED:-0}
TESTS_PASSED=${TESTS_PASSED:-0}
TESTS_FAILED=${TESTS_FAILED:-0}

# Now show output and clean up
cat "$PYTEST_OUTPUT"
rm "$PYTEST_OUTPUT"
set -e

echo "Tests collected: ${TESTS_COLLECTED}, passed: ${TESTS_PASSED}, failed: ${TESTS_FAILED}"
echo ""

# Fail on 0 tests collected (unless --allow-empty)
if [ "$TESTS_COLLECTED" -eq 0 ]; then
    if [ "$ALLOW_EMPTY" -eq 1 ]; then
        echo -e "${YELLOW}⚠ 0 tests collected (allowed via --allow-empty)${NC}"
    else
        echo -e "${RED}ERROR: 0 tests collected. Contract gate cannot pass without executing tests.${NC}"
        echo "       Use --allow-empty to permit local smoke runs."
        exit 2
    fi
elif [ $EXIT_CODE -ne 0 ] || [ "$TESTS_FAILED" -gt 0 ]; then
    echo -e "${RED}✗ Contract compliance tests failed${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Contract compliance tests passed (${TESTS_PASSED} tests)${NC}"
fi
echo ""

# Find latest run directory
echo "[4/5] Locating test output directory..."
LATEST_RUN=$(ls -td "${REPO_ROOT}/outputs/run_"* 2>/dev/null | head -1 || true)

if [ -z "${LATEST_RUN}" ]; then
    echo -e "${YELLOW}⚠ No outputs/run_* directory found${NC}"
    echo "  (This is OK if tests didn't produce output files)"
    LATEST_RUN="${REPO_ROOT}/outputs"
else
    echo -e "${GREEN}✓ Found: ${LATEST_RUN}${NC}"
fi
echo ""

# Run schema lint
echo "[5/5] Running schema lint (strict mode)..."
if [ -d "${LATEST_RUN}" ]; then
    LINT_OUTPUT=$(mktemp)
    set +e
    python3 "${REPO_ROOT}/tools/schema_lint.py" --dir "${LATEST_RUN}" --strict > "$LINT_OUTPUT" 2>&1
    LINT_EXIT=$?
    set -e
    
    cat "$LINT_OUTPUT"
    echo ""
    
    # Parse event counts
    FILES_CHECKED=0
    EVENTS_VALID=0
    
    if grep -q "Files checked:" "$LINT_OUTPUT" 2>/dev/null; then
        FILES_CHECKED=$(grep "Files checked:" "$LINT_OUTPUT" | grep -oE "[0-9]+")
    fi
    if grep -q "Events valid:" "$LINT_OUTPUT" 2>/dev/null; then
        EVENTS_VALID=$(grep "Events valid:" "$LINT_OUTPUT" | grep -oE "[0-9]+")
    fi
    
    FILES_CHECKED=${FILES_CHECKED:-0}
    EVENTS_VALID=${EVENTS_VALID:-0}
    
    rm "$LINT_OUTPUT"
    
    # In strict mode, fail on 0 events (unless --allow-empty-events)
    if [ "$EVENTS_VALID" -eq 0 ] && [ "$ALLOW_EMPTY_EVENTS" -eq 0 ]; then
        echo -e "${RED}ERROR: 0 events validated. Contract gate cannot pass without events.${NC}"
        echo "       Use --allow-empty-events to permit local smoke runs."
        exit 3
    fi
    
    # Check for actual errors in lint output
    if [ $LINT_EXIT -ne 0 ]; then
        if grep -q "Errors: 0" <<< "$(cat /dev/stdin 2>/dev/null || echo '')" 2>/dev/null; then
            # No errors, only warnings
            echo -e "${GREEN}✓ Schema lint passed (warnings only)${NC}"
        else
            echo -e "${RED}✗ Schema lint failed with errors${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✓ Schema lint passed (${FILES_CHECKED} files, ${EVENTS_VALID} events)${NC}"
    fi
else
    if [ "$ALLOW_EMPTY_EVENTS" -eq 1 ]; then
        echo -e "${YELLOW}⚠ Skipping schema lint (no output directory) --allow-empty-events${NC}"
    else
        echo -e "${RED}ERROR: No output directory found. Contract gate cannot pass without events.${NC}"
        exit 3
    fi
fi
echo ""

# Summary
echo "========================================"
echo -e "${GREEN}CONTRACT COMPLIANCE SUITE PASSED${NC}"
echo "========================================"
echo ""
echo "Summary:"
echo "  - Runtime directories cleaned"
echo "  - CGF server started and stopped"
echo "  - Contract compliance tests: ${TESTS_PASSED} PASSED"
echo "  - Output directory: ${LATEST_RUN}"
echo "  - Schema lint: ${EVENTS_VALID} events validated"
echo ""
