#!/bin/bash
# 🔐 AIPROD Environment Validation Script
# Validates that all required environment variables are correctly set
# and not exposing secrets

echo "🔐 AIPROD Environment Validation"
echo "================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
PASSED=0
FAILED=0
WARNINGS=0

# Function to check if variable is set and not placeholder
check_var() {
    local var_name=$1
    local var_value=${!var_name}
    local is_critical=${2:-false}
    
    if [ -z "$var_value" ]; then
        if [ "$is_critical" = true ]; then
            echo -e "${RED}❌ $var_name is not set (CRITICAL)${NC}"
            FAILED=$((FAILED+1))
        else
            echo -e "${YELLOW}⚠️  $var_name not configured (optional)${NC}"
            WARNINGS=$((WARNINGS+1))
        fi
    elif [[ "$var_value" == "your-"* ]] || [[ "$var_value" == "test-"* ]] || [[ "$var_value" == "path/to/"* ]]; then
        echo -e "${RED}❌ $var_name has placeholder value${NC}"
        FAILED=$((FAILED+1))
    else
        # Mask the value for display
        masked="${var_value:0:4}****...${var_value: -4}"
        echo -e "${GREEN}✅ $var_name configured${NC}"
        PASSED=$((PASSED+1))
    fi
}

# Check critical variables
echo "Critical Variables:"
echo "-------------------"
check_var "GOOGLE_CLOUD_PROJECT" true
check_var "DATABASE_URL" true
check_var "GEMINI_API_KEY" true
check_var "RUNWAY_API_KEY" true

echo ""
echo "Optional Variables:"
echo "-------------------"
check_var "REPLICATE_API_KEY" false
check_var "ELEVENLABS_API_KEY" false
check_var "FIREBASE_CREDENTIALS" false

echo ""
echo "==================================="
echo "Summary:"
echo -e "  ${GREEN}✅ Passed: ${PASSED}${NC}"
echo -e "  ${RED}❌ Failed: ${FAILED}${NC}"
echo -e "  ${YELLOW}⚠️  Warnings: ${WARNINGS}${NC}"
echo "==================================="

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Environment validation FAILED${NC}"
    exit 1
else
    echo ""
    echo -e "${GREEN}✅ Environment validation PASSED${NC}"
    exit 0
fi
