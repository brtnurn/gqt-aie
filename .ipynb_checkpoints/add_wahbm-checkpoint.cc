#include <stdint.h>
#include <aie_api/aie.hpp>

// Helper function to simulate adding bits to a specific field
// Note: In a real scenario, you might want to inline this or optimize memory access

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
    /*uint32_t aie_add_wahbm(uint32_t *R,
                           uint32_t r_size,
                           uint32_t *wah,
                           uint32_t wah_size)
    {
        constexpr int vec_factor = 32;
        
        int N = wah_size / vec_factor;

        uint32_t field_i = 0;
        uint32_t *__restrict pWah = wah;

        for(int i = 0; i < N; i++) {

            aie::vector<uint32_t, vec_factor> wah_vec = aie::load_v<vec_factor>(pWah);
            pWah += vec_factor;
            
            //uint32_t msb[vec_factor];
            //aie::store_v(msb, aie::bit_and((uint32_t)0x80000000, wah_vec));
            aie::vector<uint32_t, vec_factor> msb_vec = aie::bit_and((uint32_t)0x80000000, wah_vec);
            
            //uint32_t num_words[vec_factor];
            //aie::store_v(num_words, aie::bit_and((uint32_t)0x3fffffff, wah_vec));
            aie::vector<uint32_t, vec_factor> num_words_vec = aie::bit_and((uint32_t)0x3fffffff, wah_vec);
            
            //uint32_t fill_bit[vec_factor];
            //aie::store_v(fill_bit, aie::bit_and((uint32_t)0x40000000, wah_vec));
            aie::vector<uint32_t, vec_factor> fill_bits_vec = aie::bit_and((uint32_t)0x40000000, wah_vec);
            
            for(int j = 0; j < vec_factor; j++) {
                
                uint32_t msb = msb_vec.get(j);
                uint32_t num_word = num_words_vec.get(j);
                uint32_t fill_bit = fill_bits_vec.get(j);
                
                if(msb != 0) {
                    if(fill_bit == 0) {
                        field_i += num_word * 31;
                        if(field_i >= r_size) {
                            return r_size;
                        }
                    }
                    else {
                        uint32_t count = num_word;
                        uint32_t bits = 0xfffffffeu;
                        for(uint32_t k = 0; k < count; k++) {
                            aie_add(bits, R, field_i);
                            field_i += 31;
                            if(field_i >= r_size) {
                                return r_size;
                            }
                        }
                    }
                }
                else {
                    uint32_t bits = num_word + fill_bit;
                    if(bits != 0) {
                        aie_add(bits, R, field_i);
                    }
                    field_i += 31;
                    if(field_i >= r_size) {
                        return r_size;
                    }
                }
            }
        }
        return r_size;
    }*/


    /*uint32_t aie_add_wahbm(uint32_t *R,
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
    
        uint32_t buf, buf_empty_bits = 32;
    
        for (wah_i = 0; wah_i < wah_size; ++wah_i) {
            wah_c = wah[wah_i];
            if (wah_c & 0x80000000) { // check msb
                num_words = (wah_c & 0x3fffffff);
                //fill_bit = (wah_c>=0xC0000000?1:0);
        		fill_bit = wah_c & 0x40000000; // check 30th bit
                bits = (fill_bit?0x7FFFFFFF:0);
            } else {
                num_words = 1;
                bits = wah_c;
            }
    
            if ( (num_words > 1) && (fill_bit == 0) ) {
                // probably need to account for extra bits here
                if (buf_empty_bits < 32)
                    aie_add(buf, R, field_i);
                
                field_i += 32;
                // the empty bits were supplied by this run, so we don't want to
                // count them twice
                field_i += num_words*31 - buf_empty_bits; 
    
                buf_empty_bits = 32;
                buf = 0;
    
                if (field_i >= r_size)
                    return r_size;
            } else {
                if (bits == 0) {
    
                    if (buf_empty_bits < 32)
                        aie_add(buf, R, field_i);
                    field_i += 32 + (31 - buf_empty_bits);
    
                    buf = 0;
                    buf_empty_bits = 32;
    
                    if (field_i >= r_size)
                        return r_size;
    
                } else {
                    for (word_i = 0; word_i < num_words; ++word_i) {
                        if (buf_empty_bits == 32) {
                            if (field_i % 32 != 0) {
                                // add padding to buf that makes up for the
                                // difference, then add 32 - (field_i % 32) bits to
                                // the buff
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
    
        for (bit_i = 0; bit_i < 31; ++bit_i) {
            R[field_i] += (buf >> (31 - bit_i)) & 1;
            field_i += 1;
    
            if (field_i >= r_size)
                return r_size;
        }
    
        return r_size;
    }*/

    uint32_t aie_add_wahbm(uint32_t *R,
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
    
        uint32_t buf, buf_empty_bits = 32;
        constexpr uint32_t one = 1;
        constexpr uint32_t zero = 0;
        constexpr uint32_t vec_factor = 32;
        int N = wah_size / vec_factor;
        uint32_t *__restrict pWah = wah;

        aie::vector<uint32_t, vec_factor> zero_vec      = aie::broadcast<uint32_t, vec_factor>(0u);
        aie::vector<uint32_t, vec_factor> one_vec       = aie::broadcast<uint32_t, vec_factor>(1u);
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
                    // probably need to account for extra bits here
                    if (buf_empty_bits < 32)
                        aie_add(buf, R, field_i);
                    
                    field_i += 32;
                    // the empty bits were supplied by this run, so we don't want to
                    // count them twice
                    field_i += num_words*31 - buf_empty_bits; 
        
                    buf_empty_bits = 32;
                    buf = 0;
        
                    if (field_i >= r_size)
                        return r_size;
                } else {
                    if (bits == 0) {
        
                        if (buf_empty_bits < 32)
                            aie_add(buf, R, field_i);
                        field_i += 32 + (31 - buf_empty_bits);
        
                        buf = 0;
                        buf_empty_bits = 32;
        
                        if (field_i >= r_size)
                            return r_size;
        
                    } else {
                        for (word_i = 0; word_i < num_words; ++word_i) {
                            if (buf_empty_bits == 32) {
                                if (field_i % 32 != 0) {
                                    // add padding to buf that makes up for the
                                    // difference, then add 32 - (field_i % 32) bits to
                                    // the buff
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
    
        for (bit_i = 0; bit_i < 31; ++bit_i) {
            R[field_i] += (buf >> (31 - bit_i)) & 1;
            field_i += 1;
    
            if (field_i >= r_size)
                return r_size;
        }
    
        return r_size;
    }

}
