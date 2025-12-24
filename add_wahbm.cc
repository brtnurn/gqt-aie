#include <stdint.h>
#include <aie_api/aie.hpp>

// Compile-time constants matching Python configuration
// These MUST match add_wahbm_multi_placed.py!
// Using 7440 = 240 × 31 to avoid ObjectFifo buffer issue with last WAH word
constexpr uint32_t CONST_R_SIZE = 7440;      // num_variant = worker_output_size
constexpr uint32_t CONST_WAH_SIZE = 240;     // 7440 / 31 = exactly 240 WAH words

/**
 * AIE vectorized bit scatter - equivalent to avx_add()
 * 
 * Extracts 32 bits from 'word' and adds each to corresponding R position.
 * Uses aie::mask and aie::select for efficient bit extraction.
 */
inline void aie_add(uint32_t word, uint32_t* out, uint32_t field_i)
{        
    constexpr int vec_factor = 32;
    constexpr uint32_t zero = 0;
    constexpr uint32_t one = 1;
    
    uint32_t *__restrict pOut = out;
        
    aie::mask<32> y1 = aie::mask<32>::from_uint32(word);
    
    aie::vector<uint32_t, vec_factor> y3 = aie::select(zero, one, y1);
    y3 = aie::reverse(y3);
    
    pOut += field_i;
    
    aie::vector<uint32_t, vec_factor> aie_out = aie::load_v<vec_factor>(pOut);
    aie::store_v(pOut, aie::add(aie_out, y3));     
}


extern "C"
{  

    uint32_t aie_add_wahbm(uint32_t *R,
                           uint32_t r_size,
                           uint32_t *wah,
                           uint32_t wah_size)
    {
        // Use compile-time constants instead of passed arguments
        // (workaround for potential MLIR argument passing issues)
        const uint32_t actual_r_size = CONST_R_SIZE;
        const uint32_t actual_wah_size = CONST_WAH_SIZE;
        
        constexpr uint32_t one = 1;
        constexpr uint32_t zero = 0;
        constexpr uint32_t vec_factor = 32;
        
        // =====================================================================
        // Step 1: Zero output buffer using vectorized stores
        // =====================================================================
        aie::vector<uint32_t, vec_factor> zeros = aie::zeros<uint32_t, vec_factor>();
        constexpr uint32_t num_vec_writes = CONST_R_SIZE / vec_factor;  // 7440/32 = 232
        
        for (uint32_t i = 0; i < num_vec_writes; ++i) {
            aie::store_v(R + i * vec_factor, zeros);
        }
        // Handle remaining elements (7440 % 32 = 16)
        for (uint32_t i = num_vec_writes * vec_factor; i < actual_r_size; ++i) {
            R[i] = 0;
        }
        
        // =====================================================================
        // Step 2: WAH decompression with vectorized processing
        // =====================================================================
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
    
        uint32_t buf = 0, buf_empty_bits = 32;
        constexpr uint32_t N = CONST_WAH_SIZE / vec_factor;  // 240/32 = 7
        uint32_t *__restrict pWah = wah;

        aie::vector<uint32_t, vec_factor> zero_vec = aie::broadcast<uint32_t, vec_factor>(0u);
        aie::vector<uint32_t, vec_factor> one_vec = aie::broadcast<uint32_t, vec_factor>(1u);
        aie::vector<uint32_t, vec_factor> fill_ones_vec = aie::broadcast<uint32_t, vec_factor>(0x7fffffffu);

        for(uint32_t i = 0; i < N; i++) {
            aie::vector<uint32_t, vec_factor> wah_vec = aie::load_v<vec_factor>(pWah);
            pWah += vec_factor;
            
            aie::vector<uint32_t, vec_factor> msb_vec = aie::bit_and(0x80000000u, wah_vec);
            aie::mask<32> msb_mask = aie::neq(zero_vec, msb_vec);

            aie::vector<uint32_t, vec_factor> y1 = aie::bit_and(0x3fffffffu, wah_vec);
            aie::vector<uint32_t, vec_factor> num_words_vec = aie::select(one_vec, y1, msb_mask);
            
            aie::vector<uint32_t, vec_factor> fill_bits_vec = aie::bit_and(0x40000000u, wah_vec);
            aie::mask<32> fill_bits_mask = aie::neq(zero_vec, fill_bits_vec);

            aie::vector<uint32_t, vec_factor> y2 = aie::select(zero_vec, fill_ones_vec, fill_bits_mask);
            aie::vector<uint32_t, vec_factor> bits_vec = aie::select(wah_vec, y2, msb_mask);
        

            for(uint32_t j = 0; j < vec_factor; j++) {
                
                bits = bits_vec.get(j);
                num_words = num_words_vec.get(j);
                fill_bit = fill_bits_vec.get(j);

                if ( (num_words > 1) && (fill_bit == 0) ) {
                    // Multi-word zero-fill: flush buffer and skip
                    if (buf_empty_bits < 32)
                        aie_add(buf, R, field_i);
                    
                    field_i += 32;
                    field_i += num_words*31 - buf_empty_bits; 
        
                    buf_empty_bits = 32;
                    buf = 0;
        
                    if (field_i >= actual_r_size)
                        return actual_r_size;
                } else {
                    if (bits == 0) {
                        // Zero literal: flush buffer and skip
                        if (buf_empty_bits < 32)
                            aie_add(buf, R, field_i);
                        field_i += 32 + (31 - buf_empty_bits);
        
                        buf = 0;
                        buf_empty_bits = 32;
        
                        if (field_i >= actual_r_size)
                            return actual_r_size;
        
                    } else {
                        // Non-zero literal or fill: buffer and process
                        for (word_i = 0; word_i < num_words; ++word_i) {
                            if (buf_empty_bits == 32) {
                                if (field_i % 32 != 0) {
                                    uint32_t padding = field_i % 32;
                                    buf = bits >> (padding - 1);
                                    aie_add(buf, R, field_i - padding);
                                    field_i+= 32 - padding;
                                    buf = bits << (32 - padding + 1);
                                    buf_empty_bits = (32 - padding) + 1;
                                } else {
                                    buf = bits << 1;
                                    buf_empty_bits = 1;
                                }
                            } else {
                                buf += bits >> (31-buf_empty_bits);
                                aie_add(buf, R, field_i);
                                field_i+=32;
                                buf_empty_bits += 1;
                                buf = bits << buf_empty_bits;
                            }
                        }
                    }
                }
            }
        }
        
        // =====================================================================
        // Step 3: Process remaining WAH words (240 % 32 = 16) with scalar decode
        // =====================================================================
        for (wah_i = N * vec_factor; wah_i < actual_wah_size; ++wah_i) {
            wah_c = wah[wah_i];
            
            if (wah_c >> 31 == 1) {
                num_words = (wah_c & 0x3fffffffu);
                fill_bit = (wah_c >= 0xC0000000u) ? 1 : 0;
                bits = fill_bit ? 0x7FFFFFFFu : 0;
            } else {
                num_words = 1;
                bits = wah_c;
            }

            if ((num_words > 1) && (fill_bit == 0)) {
                if (buf_empty_bits < 32)
                    aie_add(buf, R, field_i);
                field_i += 32;
                field_i += num_words * 31 - buf_empty_bits;
                buf_empty_bits = 32;
                buf = 0;
                if (field_i >= actual_r_size)
                    return actual_r_size;
            } else if (bits == 0) {
                if (buf_empty_bits < 32)
                    aie_add(buf, R, field_i);
                field_i += 32 + (31 - buf_empty_bits);
                buf = 0;
                buf_empty_bits = 32;
                if (field_i >= actual_r_size)
                    return actual_r_size;
            } else {
                for (word_i = 0; word_i < num_words; ++word_i) {
                    if (buf_empty_bits == 32) {
                        if (field_i % 32 != 0) {
                            uint32_t padding = field_i % 32;
                            buf = bits >> (padding - 1);
                            aie_add(buf, R, field_i - padding);
                            field_i += 32 - padding;
                            buf = bits << (32 - padding + 1);
                            buf_empty_bits = (32 - padding) + 1;
                        } else {
                            buf = bits << 1;
                            buf_empty_bits = 1;
                        }
                    } else {
                        buf += bits >> (31 - buf_empty_bits);
                        aie_add(buf, R, field_i);
                        field_i += 32;
                        buf_empty_bits += 1;
                        buf = bits << buf_empty_bits;
                    }
                }
            }
        }
    
        // =====================================================================
        // Step 4: Flush remaining bits in buffer
        // =====================================================================
        for (bit_i = 0; bit_i < 31; ++bit_i) {
            R[field_i] += (buf >> (31 - bit_i)) & 1;
            field_i += 1;
    
            if (field_i >= actual_r_size)
                return actual_r_size;
        }
    
        return actual_r_size;
    }


}
