#!/bin/bash
# Test script to find the failure threshold for batch processing

OUTPUT_FILE="test_results.txt"
echo "=== Batch Processing Test Results ===" > $OUTPUT_FILE
echo "Date: $(date)" >> $OUTPUT_FILE
echo "" >> $OUTPUT_FILE

# Function to run a test and log result
run_test() {
    local samples=$1
    local variants=$2
    local sample_batches=$(( (samples + 31) / 32 ))
    local variant_batches=$(( (variants + 7439) / 7440 ))
    local total_calls=$((sample_batches * variant_batches))
    
    echo "Testing SAMPLES=$samples VARIANTS=$variants (${sample_batches}×${variant_batches}=${total_calls} NPU calls)..."
    echo "----------------------------------------" >> $OUTPUT_FILE
    echo "SAMPLES=$samples VARIANTS=$variants" >> $OUTPUT_FILE
    echo "Sample batches: $sample_batches, Variant batches: $variant_batches, Total NPU calls: $total_calls" >> $OUTPUT_FILE
    
    # Run the test and capture output
    output=$(make run use_placed=1 SAMPLES=$samples VARIANTS=$variants 2>&1)
    
    # Check if PASS or FAIL
    if echo "$output" | grep -q "PASS"; then
        echo "  -> PASS" 
        echo "RESULT: PASS" >> $OUTPUT_FILE
    else
        echo "  -> FAIL"
        echo "RESULT: FAIL" >> $OUTPUT_FILE
        # Extract error count
        error_count=$(echo "$output" | grep "mismatches" | head -1)
        if [ -n "$error_count" ]; then
            echo "  $error_count" >> $OUTPUT_FILE
        fi
        # Extract first few errors
        echo "$output" | grep "Error at variant" | head -5 >> $OUTPUT_FILE
    fi
    echo "" >> $OUTPUT_FILE
}

echo "Starting tests..."
echo ""

# =============================================================================
# Test 1: Vary samples with small variant count (1 variant batch)
# =============================================================================
echo "=== Test Group 1: Varying samples, 1 variant batch (7471 variants) ===" >> $OUTPUT_FILE
for samples in 32 64 96 128; do
    run_test $samples 7471
done

# =============================================================================
# Test 2: Vary samples with 2 variant batches
# =============================================================================
echo "=== Test Group 2: Varying samples, 2 variant batches (14942 variants) ===" >> $OUTPUT_FILE
for samples in 32 64 96 128; do
    run_test $samples 14942
done

# =============================================================================
# Test 3: Fixed 32 samples, vary variant batches
# =============================================================================
echo "=== Test Group 3: 32 samples, varying variant batches ===" >> $OUTPUT_FILE
for variants in 7471 14942 22413 29884 37355 44826 52297 59768; do
    run_test 32 $variants
done

# =============================================================================
# Test 4: Find the exact threshold for 32 samples
# =============================================================================
echo "=== Test Group 4: 32 samples, fine-grained variant search ===" >> $OUTPUT_FILE
# If 22413 passes and something higher fails, narrow down
for variants in 25000 27000 29000 29884; do
    run_test 32 $variants
done

# =============================================================================
# Test 5: Total NPU calls threshold (different combinations, same total)
# =============================================================================
echo "=== Test Group 5: Same total NPU calls, different configurations ===" >> $OUTPUT_FILE
# 4 total calls
run_test 32 29884   # 1 × 4 = 4 calls
run_test 64 14942   # 2 × 2 = 4 calls
run_test 128 7471   # 4 × 1 = 4 calls

# 6 total calls
run_test 32 44826   # 1 × 6 = 6 calls
run_test 64 22413   # 2 × 3 = 6 calls
run_test 96 14942   # 3 × 2 = 6 calls

# =============================================================================
# Test 6: Very small tests (should all pass)
# =============================================================================
echo "=== Test Group 6: Minimal tests (baseline) ===" >> $OUTPUT_FILE
run_test 32 7471    # 1 × 1 = 1 call
run_test 32 14942   # 1 × 2 = 2 calls
run_test 64 7471    # 2 × 1 = 2 calls

echo ""
echo "==============================================================" 
echo "Tests complete! Results saved to $OUTPUT_FILE"
echo "=============================================================="
echo ""
cat $OUTPUT_FILE

