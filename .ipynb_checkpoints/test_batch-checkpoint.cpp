//===- test_batch.cpp - Batch processing for large datasets ---------------===//
//
// Demonstrates how to process N samples × M variants using batch processing
// when constrained by:
//   - 32 AIE tiles (process 32 samples in parallel)
//   - ~7500 variants per batch (memory constraint per tile)
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cxxopts.hpp"
#include "test_utils.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using DATATYPE = std::uint32_t;
using namespace std::chrono;

// ============================================================================
// Configuration Constants
// ============================================================================
constexpr int WORKER_PER_COL = 4;
constexpr int COL_COUNT = 8;
constexpr int NUM_WORKERS = WORKER_PER_COL * COL_COUNT;  // 32 AIE tiles

// Per-batch constraints (must match add_wahbm_multi_placed.py and add_wahbm.cc)
// Using 7440 = 240 × 31 to avoid ObjectFifo buffer issue with last WAH word
// (positions 7441+ showed corruption after 4+ NPU invocations with 7471)
constexpr int VARIANTS_PER_BATCH = 7440;
constexpr int WAH_WORDS_PER_WORKER = 240;  // Exact: 7440 / 31 = 240

// Buffer sizes for NPU
constexpr int NPU_IN_SIZE = WAH_WORDS_PER_WORKER * NUM_WORKERS;
constexpr int NPU_OUT_SIZE = VARIANTS_PER_BATCH * NUM_WORKERS;

// ============================================================================
// WAH Data Generation (simulating real GQT data)
// ============================================================================
void generate_wah_for_sample_variant_range(
    uint32_t* wah,
    int sample_id,
    int variant_start,
    int variant_count,
    float density = 0.05)
{
    // In real GQT, you would:
    // 1. Open the .gqt index file
    // 2. Seek to the WAH bitmap for sample_id, genotype type (HET/HOM_ALT/etc)
    // 3. Read the WAH words covering variants [variant_start, variant_start + variant_count)
    
    // For this demo, we generate random WAH data
    srand(sample_id * 10000 + variant_start);  // Reproducible per sample+variant
    
    int wah_size = (variant_count + 30) / 31;
    for (int i = 0; i < wah_size; i++) {
        uint32_t literal = 0;
        for (int b = 0; b < 31; b++) {
            if ((float)rand() / RAND_MAX < density) {
                literal |= (1u << (30 - b));
            }
        }
        wah[i] = literal;
    }
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================
uint32_t add_wahbm(uint32_t *R, uint32_t r_size, uint32_t *wah, uint32_t wah_size)
{
    uint32_t wah_c, num_words, fill_bit, bits, bit_i, word_i, field_i = 0;

    for (uint32_t wah_i = 0; wah_i < wah_size; ++wah_i) {
        wah_c = wah[wah_i];
        if (wah_c >> 31 == 1) {
            num_words = (wah_c & 0x3fffffff);
            fill_bit = (wah_c >= 0xC0000000) ? 1 : 0;
            bits = (fill_bit ? 0x7FFFFFFF : 0);
        } else {
            num_words = 1;
            bits = wah_c;
        }

        if ((num_words > 1) && (fill_bit == 0)) {
            field_i += num_words * 31;
            if (field_i >= r_size) return r_size;
        } else {
            if (bits == 0) {
                field_i += 31;
                if (field_i >= r_size) return r_size;
            } else {
                for (word_i = 0; word_i < num_words; ++word_i) {
                    for (bit_i = 0; bit_i < 31; ++bit_i) {
                        R[field_i] += (bits >> (30 - bit_i)) & 1;
                        field_i += 1;
                        if (field_i >= r_size) return r_size;
                    }
                }
            }
        }
    }
    return r_size;
}

// ============================================================================
// Host-side Sum Reduction
// ============================================================================
// NOTE: worker_stride is the actual spacing between workers in the output buffer
//       (always VARIANTS_PER_BATCH), while element_count is how many to read
//       (may be smaller for partial batches)
void host_sum_reduction(const uint32_t* worker_outputs, 
                        uint32_t* result, 
                        size_t num_workers, 
                        size_t worker_stride,
                        size_t element_count) 
{
    std::memset(result, 0, element_count * sizeof(uint32_t));
    for (size_t w = 0; w < num_workers; ++w) {
        size_t offset = w * worker_stride;  // Use fixed stride, not element_count!
        for (size_t i = 0; i < element_count; ++i) {
            result[i] += worker_outputs[offset + i];
        }
    }
}

// ============================================================================
// Main: Batch Processing Demo
// ============================================================================
int main(int argc, const char *argv[])
{
    // -------------------------------------------------------------------------
    // Parse Arguments
    // -------------------------------------------------------------------------
    cxxopts::Options options("batch-processing");
    test_utils::add_default_options(options);
    options.add_options()
        ("total-samples", "Total number of samples to process", 
         cxxopts::value<int>()->default_value("64"))
        ("total-variants", "Total number of variants to process", 
         cxxopts::value<int>()->default_value("8192"));

    cxxopts::ParseResult vm;
    test_utils::parse_options(argc, argv, options, vm);
    int verbosity = vm["verbosity"].as<int>();
    
    // -------------------------------------------------------------------------
    // Dataset Configuration
    // -------------------------------------------------------------------------
    const int TOTAL_SAMPLES = vm["total-samples"].as<int>();
    const int TOTAL_VARIANTS = vm["total-variants"].as<int>();
    
    // Calculate number of batches
    const int SAMPLE_BATCHES = (TOTAL_SAMPLES + NUM_WORKERS - 1) / NUM_WORKERS;
    const int VARIANT_BATCHES = (TOTAL_VARIANTS + VARIANTS_PER_BATCH - 1) / VARIANTS_PER_BATCH;
    
    std::cout << "=== Batch Processing Configuration ===" << std::endl;
    std::cout << "Total samples:    " << TOTAL_SAMPLES << std::endl;
    std::cout << "Total variants:   " << TOTAL_VARIANTS << std::endl;
    std::cout << "Sample batches:   " << SAMPLE_BATCHES << " (32 samples each)" << std::endl;
    std::cout << "Variant batches:  " << VARIANT_BATCHES << " (" << VARIANTS_PER_BATCH << " variants each)" << std::endl;
    std::cout << "Total NPU calls:  " << SAMPLE_BATCHES * VARIANT_BATCHES << std::endl;
    std::cout << std::endl;

    // -------------------------------------------------------------------------
    // Initialize XRT
    // -------------------------------------------------------------------------
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(vm["instr"].as<std::string>());

    xrt::device device;
    xrt::kernel kernel;
    test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                     vm["xclbin"].as<std::string>(),
                                     vm["kernel"].as<std::string>());

    // -------------------------------------------------------------------------
    // Create Buffer Objects (reused across batches)
    // -------------------------------------------------------------------------
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_inA = xrt::bo(device, NPU_IN_SIZE * sizeof(DATATYPE),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_outC = xrt::bo(device, NPU_OUT_SIZE * sizeof(DATATYPE),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    // Copy instruction stream (once)
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Map buffers once (avoid repeated mapping in loop)
    DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
    DATATYPE *bufOut = bo_outC.map<DATATYPE *>();

    // -------------------------------------------------------------------------
    // Allocate Global Result Array
    // -------------------------------------------------------------------------
    std::vector<uint32_t> global_result_npu(TOTAL_VARIANTS, 0);
    std::vector<uint32_t> global_result_cpu(TOTAL_VARIANTS, 0);
    
    // Temporary buffers
    std::vector<uint32_t> wah_batch(NPU_IN_SIZE);
    std::vector<uint32_t> batch_result(VARIANTS_PER_BATCH);

    // -------------------------------------------------------------------------
    // Batch Processing Loop
    // -------------------------------------------------------------------------
    auto total_start = high_resolution_clock::now();
    long total_npu_kernel_us = 0;
    long total_npu_sync_to_device_us = 0;
    long total_npu_sync_from_device_us = 0;
    long total_host_sum_reduction_us = 0;
    long total_global_accumulate_us = 0;
    long total_data_prep_us = 0;
    
    for (int sb = 0; sb < SAMPLE_BATCHES; ++sb) {
        int sample_start = sb * NUM_WORKERS;
        int samples_in_batch = std::min(NUM_WORKERS, TOTAL_SAMPLES - sample_start);
        
        for (int vb = 0; vb < VARIANT_BATCHES; ++vb) {
            int variant_start = vb * VARIANTS_PER_BATCH;
            int variants_in_batch = std::min(VARIANTS_PER_BATCH, TOTAL_VARIANTS - variant_start);
            
            if (verbosity >= 1) {
                std::cout << "Processing: samples [" << sample_start << "-" 
                          << sample_start + samples_in_batch - 1 << "], variants ["
                          << variant_start << "-" << variant_start + variants_in_batch - 1 
                          << "]" << std::endl;
            }
            
            // -----------------------------------------------------------------
            // Step 1: Prepare WAH data for this batch
            // -----------------------------------------------------------------
            auto prep_start = high_resolution_clock::now();
            
            std::memset(wah_batch.data(), 0, NPU_IN_SIZE * sizeof(uint32_t));
            
            for (int w = 0; w < NUM_WORKERS; ++w) {
                int sample_id = sample_start + w;
                if (sample_id < TOTAL_SAMPLES) {
                    generate_wah_for_sample_variant_range(
                        &wah_batch[w * WAH_WORDS_PER_WORKER],
                        sample_id,
                        variant_start,
                        variants_in_batch
                    );
                }
            }
            
            memcpy(bufInA, wah_batch.data(), NPU_IN_SIZE * sizeof(uint32_t));
            memset(bufOut, 0, NPU_OUT_SIZE * sizeof(DATATYPE));
            
            auto prep_stop = high_resolution_clock::now();
            total_data_prep_us += duration_cast<microseconds>(prep_stop - prep_start).count();
            
            // -----------------------------------------------------------------
            // Step 2: Execute NPU kernel
            // -----------------------------------------------------------------
            auto sync_to_start = high_resolution_clock::now();
            bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto sync_to_stop = high_resolution_clock::now();
            total_npu_sync_to_device_us += duration_cast<microseconds>(sync_to_stop - sync_to_start).count();
            
            auto kernel_start = high_resolution_clock::now();
            auto run = kernel(3, bo_instr, instr_v.size(), bo_inA, bo_outC);
            run.wait();
            auto kernel_stop = high_resolution_clock::now();
            total_npu_kernel_us += duration_cast<microseconds>(kernel_stop - kernel_start).count();
            
            auto sync_from_start = high_resolution_clock::now();
            bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            auto sync_from_stop = high_resolution_clock::now();
            total_npu_sync_from_device_us += duration_cast<microseconds>(sync_from_stop - sync_from_start).count();
            
            // -----------------------------------------------------------------
            // Step 3: Sum reduction across workers
            // -----------------------------------------------------------------
            auto reduction_start = high_resolution_clock::now();
            host_sum_reduction(bufOut, batch_result.data(), NUM_WORKERS, VARIANTS_PER_BATCH, variants_in_batch);
            auto reduction_stop = high_resolution_clock::now();
            total_host_sum_reduction_us += duration_cast<microseconds>(reduction_stop - reduction_start).count();
            
            // -----------------------------------------------------------------
            // Step 4: Accumulate into global result
            // -----------------------------------------------------------------
            auto accum_start = high_resolution_clock::now();
            for (int i = 0; i < variants_in_batch; ++i) {
                global_result_npu[variant_start + i] += batch_result[i];
            }
            auto accum_stop = high_resolution_clock::now();
            total_global_accumulate_us += duration_cast<microseconds>(accum_stop - accum_start).count();
        }
    }
    
    auto total_stop = high_resolution_clock::now();
    auto npu_total_time = duration_cast<microseconds>(total_stop - total_start);

    // -------------------------------------------------------------------------
    // CPU Reference Timing (separate, fair comparison)
    // -------------------------------------------------------------------------
    std::cout << "Running CPU reference for timing comparison..." << std::endl;
    
    auto cpu_start = high_resolution_clock::now();
    long total_cpu_data_prep_us = 0;
    long total_cpu_add_wahbm_us = 0;
    
    for (int sb = 0; sb < SAMPLE_BATCHES; ++sb) {
        int sample_start = sb * NUM_WORKERS;
        int samples_in_batch = std::min(NUM_WORKERS, TOTAL_SAMPLES - sample_start);
        
        for (int vb = 0; vb < VARIANT_BATCHES; ++vb) {
            int variant_start = vb * VARIANTS_PER_BATCH;
            int variants_in_batch = std::min(VARIANTS_PER_BATCH, TOTAL_VARIANTS - variant_start);
            
            // Data preparation (same as NPU)
            auto cpu_prep_start = high_resolution_clock::now();
            std::memset(wah_batch.data(), 0, NPU_IN_SIZE * sizeof(uint32_t));
            
            for (int w = 0; w < NUM_WORKERS; ++w) {
                int sample_id = sample_start + w;
                if (sample_id < TOTAL_SAMPLES) {
                    generate_wah_for_sample_variant_range(
                        &wah_batch[w * WAH_WORDS_PER_WORKER],
                        sample_id,
                        variant_start,
                        variants_in_batch
                    );
                }
            }
            auto cpu_prep_stop = high_resolution_clock::now();
            total_cpu_data_prep_us += duration_cast<microseconds>(cpu_prep_stop - cpu_prep_start).count();
            
            // CPU add_wahbm processing
            auto cpu_wahbm_start = high_resolution_clock::now();
            int actual_wah_words = (variants_in_batch + 30) / 31;
            for (int w = 0; w < samples_in_batch; ++w) {
                add_wahbm(&global_result_cpu[variant_start], variants_in_batch,
                         &wah_batch[w * WAH_WORDS_PER_WORKER], actual_wah_words);
            }
            auto cpu_wahbm_stop = high_resolution_clock::now();
            total_cpu_add_wahbm_us += duration_cast<microseconds>(cpu_wahbm_stop - cpu_wahbm_start).count();
        }
    }
    
    auto cpu_stop = high_resolution_clock::now();
    auto cpu_total_time = duration_cast<microseconds>(cpu_stop - cpu_start);

    // -------------------------------------------------------------------------
    // Verification
    // -------------------------------------------------------------------------
    int errors = 0;
    for (int i = 0; i < TOTAL_VARIANTS; ++i) {
        if (global_result_npu[i] != global_result_cpu[i]) {
            if (errors < 10) {
                std::cout << "Error at variant " << i << ": NPU=" 
                          << global_result_npu[i] << " CPU=" 
                          << global_result_cpu[i] << std::endl;
            }
            errors++;
        }
    }

    // -------------------------------------------------------------------------
    // Results
    // -------------------------------------------------------------------------
    std::cout << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "           PERFORMANCE COMPARISON           " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Dataset: " << TOTAL_SAMPLES << " samples x " << TOTAL_VARIANTS << " variants" << std::endl;
    std::cout << "NPU batches: " << SAMPLE_BATCHES * VARIANT_BATCHES << " calls" << std::endl;
    std::cout << std::endl;
    
    // Calculate totals
    long npu_compute_total = total_npu_kernel_us + total_host_sum_reduction_us + total_global_accumulate_us;
    long npu_sync_total = total_npu_sync_to_device_us + total_npu_sync_from_device_us;
    
    std::cout << "--- NPU Pipeline Breakdown ---" << std::endl;
    std::cout << "  Data preparation:      " << total_data_prep_us << " us" << std::endl;
    std::cout << "  Sync to device:        " << total_npu_sync_to_device_us << " us" << std::endl;
    std::cout << "  Kernel execution:      " << total_npu_kernel_us << " us" << std::endl;
    std::cout << "  Sync from device:      " << total_npu_sync_from_device_us << " us" << std::endl;
    std::cout << "  Host sum reduction:    " << total_host_sum_reduction_us << " us" << std::endl;
    std::cout << "  Global accumulation:   " << total_global_accumulate_us << " us" << std::endl;
    std::cout << "  --------------------------" << std::endl;
    std::cout << "  Total wall time:       " << npu_total_time.count() << " us" << std::endl;
    std::cout << std::endl;
    
    std::cout << "--- CPU Pipeline Breakdown ---" << std::endl;
    std::cout << "  Data preparation:      " << total_cpu_data_prep_us << " us" << std::endl;
    std::cout << "  add_wahbm processing:  " << total_cpu_add_wahbm_us << " us" << std::endl;
    std::cout << "  --------------------------" << std::endl;
    std::cout << "  Total wall time:       " << cpu_total_time.count() << " us" << std::endl;
    std::cout << std::endl;
    
    // Speedup calculations
    float speedup_kernel_vs_wahbm = (float)total_cpu_add_wahbm_us / (float)total_npu_kernel_us;
    float speedup_compute = (float)total_cpu_add_wahbm_us / (float)npu_compute_total;
    float speedup_total = (float)cpu_total_time.count() / (float)npu_total_time.count();
    
    std::cout << "--- Speedup Analysis ---" << std::endl;
    std::cout << "  NPU kernel vs CPU add_wahbm:     " << speedup_kernel_vs_wahbm << "x" << std::endl;
    std::cout << "  NPU compute vs CPU add_wahbm:    " << speedup_compute << "x" << std::endl;
    std::cout << "    (kernel + sum_reduction + accumulation)" << std::endl;
    std::cout << "  NPU total vs CPU total:          " << speedup_total << "x" << std::endl;
    std::cout << std::endl;
    
    // Breakdown percentages
    std::cout << "--- NPU Time Distribution ---" << std::endl;
    float total_time = (float)npu_total_time.count();
    std::cout << "  Data prep:     " << (100.0f * total_data_prep_us / total_time) << "%" << std::endl;
    std::cout << "  Sync to dev:   " << (100.0f * total_npu_sync_to_device_us / total_time) << "%" << std::endl;
    std::cout << "  Kernel:        " << (100.0f * total_npu_kernel_us / total_time) << "%" << std::endl;
    std::cout << "  Sync from dev: " << (100.0f * total_npu_sync_from_device_us / total_time) << "%" << std::endl;
    std::cout << "  Sum reduction: " << (100.0f * total_host_sum_reduction_us / total_time) << "%" << std::endl;
    std::cout << "  Accumulation:  " << (100.0f * total_global_accumulate_us / total_time) << "%" << std::endl;
    std::cout << std::endl;
    
    std::cout << "============================================" << std::endl;

    if (errors == 0) {
        std::cout << "PASS! All " << TOTAL_VARIANTS << " variants verified." << std::endl;
        return 0;
    } else {
        std::cout << errors << " mismatches out of " << TOTAL_VARIANTS << " variants." << std::endl;
        std::cout << "FAIL." << std::endl;
        return 1;
    }
}

