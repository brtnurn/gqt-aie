//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
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

#ifndef DATATYPES_USING_DEFINED
#define DATATYPES_USING_DEFINED
using DATATYPE = std::uint32_t; // Configure this to match your buffer data type
#endif
using namespace std::chrono;

const int scaleFactor = 3;

uint32_t add_wahbm(uint32_t *R,
                   uint32_t r_size,
                   uint32_t *wah,
                   uint32_t wah_size)
{

    uint32_t wah_c,
             wah_i,
             num_words,
             fill_bit,
             bits,
             bit,
             bit_i,
             word_i,
             field_i;

    field_i = 0;

    uint32_t v;

    for (wah_i = 0; wah_i < wah_size; ++wah_i) {
        wah_c = wah[wah_i];
        if (wah_c >> 31 == 1) {
            num_words = (wah_c & 0x3fffffff);
            fill_bit = (wah_c>=0xC0000000?1:0);
            bits = (fill_bit?0x7FFFFFFF:0);
        } else {
            num_words = 1;
            bits = wah_c;
        }

        if ( (num_words > 1) && (fill_bit == 0) ) {
            field_i += num_words * 31;
            if (field_i >= r_size)
                return r_size;
        } else {
            if (bits == 0) {
                field_i += 31;
                if (field_i >= r_size)
                    return r_size;
            } else {
                for (word_i = 0; word_i < num_words; ++word_i) {
                    /* 
                    // Attempt to reduce the number of times the for loop
                    // itterates so that 
                    v = bits;
                    for ( ; v ; ) {
                        R[field_i] += log2_32(v&(-v));
                        v &= v - 1;
                        field_i += 1;
                        if (field_i >= r_size)
                            return r_size;
                    }
                    */
                    for (bit_i = 0; bit_i < 31; ++bit_i) {
                        R[field_i] += (bits >> (30 - bit_i)) & 1;
                        field_i += 1;

                        if (field_i >= r_size)
                            return r_size;
                    }
                }
            }
        }
    }

    return r_size;
}

void host_sum_reduction(const uint32_t* input_buffer, 
                        uint32_t* output_buffer, 
                        size_t num_workers, 
                        size_t element_count) {
    
    // 1. Initialize output buffer to 0
    //    Using memset for speed (0x00 bytes result in integer 0)
    std::memset(output_buffer, 0, element_count * sizeof(uint32_t));

    // 2. Accumulate
    //    We loop through workers, then elements. This accesses 'input_buffer'
    //    sequentially, which is friendly to the CPU cache prefetcher.
    for (size_t w = 0; w < num_workers; ++w) {
        
        // Calculate the starting offset for this specific worker's data
        size_t offset = w * element_count;
        
        for (size_t i = 0; i < element_count; ++i) {
            output_buffer[i] += input_buffer[offset + i];
        }
    }
}

int main(int argc, const char *argv[])
{
	srand(time(NULL));
    // Program arguments parsing
    cxxopts::Options options("section-3");
    test_utils::add_default_options(options);

    cxxopts::ParseResult vm;
    test_utils::parse_options(argc, argv, options, vm);
    int verbosity = vm["verbosity"].as<int>();

    // Declaring design constants
    constexpr bool VERIFY = true;
    constexpr int WORKER_PER_COL = 4;
    constexpr int COL_COUNT = 8;
    constexpr int NUM_WORKERS = WORKER_PER_COL * COL_COUNT;
    constexpr int R_SIZE = 8192;
    constexpr int OUT_SIZE = R_SIZE * NUM_WORKERS;
    constexpr int IN_SIZE = ((R_SIZE + 31 - 1) / 31) * NUM_WORKERS;

    // Load instruction sequence
    std::vector<uint32_t> instr_v =
        test_utils::load_instr_binary(vm["instr"].as<std::string>());

    if (verbosity >= 1)
        std::cout << "Sequence instr count: " << instr_v.size() << "\n";

    // Start the XRT context and load the kernel
    xrt::device device;
    xrt::kernel kernel;

    test_utils::init_xrt_load_kernel(device, kernel, verbosity,
                                     vm["xclbin"].as<std::string>(),
                                     vm["kernel"].as<std::string>());

    // set up the buffer objects
    auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                            XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_inA = xrt::bo(device, IN_SIZE * sizeof(DATATYPE),
                          XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_outC = xrt::bo(device, OUT_SIZE * sizeof(DATATYPE),
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

    if (verbosity >= 1)
        std::cout << "Writing data into buffer objects.\n";

    // Copy instruction stream to xrt buffer object
    void *bufInstr = bo_instr.map<void *>();
    memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

    uint32_t R[R_SIZE] = {0};
    uint32_t wah[IN_SIZE];
    
    // Initialize buffer bo_inA
    DATATYPE *bufInA = bo_inA.map<DATATYPE *>();
    for (int i = 0; i < IN_SIZE; i++) {
        uint32_t n = rand();
        bufInA[i] = n;
        wah[i] = n;
    }
    
    // Zero out buffer bo_outC
    DATATYPE *bufOut = bo_outC.map<DATATYPE *>();
    memset(bufOut, 0, OUT_SIZE * sizeof(DATATYPE));

    // sync host to device memories
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_outC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Execute the kernel and wait to finish
    if (verbosity >= 1)
        std::cout << "Running Kernel.\n";
    unsigned int opcode = 3;

    auto start = high_resolution_clock::now();
    
    auto run =
        kernel(opcode, bo_instr, instr_v.size(), bo_inA, bo_outC);
    run.wait();
    
    // Sync device to host memories
    bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);


    uint32_t out[R_SIZE] = {0};
    host_sum_reduction(bufOut, 
                       out, 
                       NUM_WORKERS, 
                       R_SIZE);
    
    auto stop = high_resolution_clock::now();
    auto npu = duration_cast<microseconds>(stop - start);
    
    // Compare out to golden
    int errors = 0;
    if (verbosity >= 1)
    {
        std::cout << "Verifying results ..." << std::endl;
    }

    const int PART_SIZE = IN_SIZE / NUM_WORKERS;
    
    start = high_resolution_clock::now();

    for(int part = 0; part < NUM_WORKERS; ++part) {
        add_wahbm(R, R_SIZE, &wah[part * PART_SIZE], PART_SIZE);
    }
    
    stop = high_resolution_clock::now();
    auto cpu = duration_cast<microseconds>(stop - start);
    
    for (uint32_t i = 0; i < R_SIZE; i++)
    {
            
        int32_t ref = R[i];
        int32_t test = out[i];
        
        if (test != ref)
        {
            if (verbosity >= 1)
                std::cout << "Error in output " << test << " != " << ref << std::endl;
            errors++;
        }
        else
        {
            if (verbosity >= 1)
                std::cout << "Correct output " << test << " == " << ref << std::endl;
        }
    }

    // Print Pass/Fail result of our test
    if (!errors)
    {
        std::cout << std::endl
                  << "PASS!" << std::endl
                  << std::endl;
        std::cout << "Time taken by NPU function: "
         << npu.count() << " microseconds" << std::endl;
        std::cout << "Time taken by CPU function: "
         << cpu.count() << " microseconds" << std::endl;
        return 0;
    }
    else
    {
        std::cout << std::endl
                  << errors << " mismatches." << std::endl
                  << std::endl;
        std::cout << std::endl
                  << "fail." << std::endl
                  << std::endl;
        return 1;
    }
}
