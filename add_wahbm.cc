#include <stdint.h>
#include <aie_api/aie.hpp>

extern "C"
{

    void aie_add(uint32_t word, uint32_t* out, uint32_t field_i)
    {
        event0();
        
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
        
        event1();
    }

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
    /*
                    
                    if (buf_empty_bits < 32)
                        avx_add(buf, &s_1, &s_2, &s_3, &s_4, &m, R_avx, field_i);
                    field_i += 32;
    
                    buf = 0;
                    buf_empty_bits = 32 - buf_empty_bits;
    */
    
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
    }
}
