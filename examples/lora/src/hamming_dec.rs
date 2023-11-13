use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use std::cmp::min;
use std::collections::{HashMap, VecDeque};
use std::f32::consts::PI;
use std::mem;
// use futuresdr::futures::FutureExt;
use futuresdr::log::{info, warn};
use futuresdr::macros::message_handler;
use futuresdr::num_complex::{Complex32, Complex64};
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::Tag;
use futuresdr::runtime::WorkIo;
use futuresdr::runtime::{Block, ItemTag};

use crate::utilities::*;

pub struct HammingDec {
    m_cr: usize, // Transmission coding rate
    // cr_app: usize,         // Coding rate use for the block
    is_header: bool,       // Indicate that it is the first block
    m_soft_decoding: bool, // Hard/Soft decoding
}

impl HammingDec {
    pub fn new(soft_decoding: bool) -> Block {
        let mut sio = StreamIoBuilder::new();
        if soft_decoding {
            sio = sio.add_input::<[LLR; 8]>("in"); // In reality: cw_len = cr_app + 4  < 8
        } else {
            sio = sio.add_input::<u8>("in");
        }
        sio = sio.add_output::<u8>("out");
        Block::new(
            BlockMetaBuilder::new("HammingDec").build(),
            sio.build(),
            MessageIoBuilder::new().build(),
            HammingDec {
                m_soft_decoding: soft_decoding,
                is_header: false,
                m_cr: 1, // TODO
            },
        )
        // set_tag_propagation_policy(TPP_ONE_TO_ONE);  // TODO
    }
}

#[async_trait]
impl Kernel for HammingDec {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let n_input = if self.m_soft_decoding {
            sio.input(0).slice::<[LLR; 8]>().len()
        } else {
            sio.input(0).slice::<u8>().len()
        };
        let output = sio.output(0).slice::<u8>();
        let mut nitems_to_process: usize = n_input;
        let mut tags: VecDeque<(usize, HashMap<String, Pmt>)> = sio
            .input(0)
            .tags()
            .iter()
            .filter_map(|x| match x {
                ItemTag {
                    index,
                    tag: Tag::NamedAny(n, val),
                } => {
                    if n == "frame_info" {
                        match (**val).downcast_ref().unwrap() {
                            Pmt::MapStrPmt(map) => Some((*index, map.clone())),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        if tags.len() > 0 {
            if tags[0].0 != 0 {
                nitems_to_process = tags[0].0; // only decode codewords until the next frame begin
            } else {
                if tags.len() >= 2 {
                    nitems_to_process = tags[1].0 - tags[0].0;
                } else {
                    nitems_to_process = n_input;
                }
                //
                //                     pmt::pmt_t err = pmt::string_to_symbol("error");
                self.is_header = if let Pmt::Bool(tmp) = tags[0].1.get("is_header").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                if !self.is_header {
                    self.m_cr = if let Pmt::Usize(tmp) = tags[0].1.get("cr").unwrap() {
                        *tmp
                    } else {
                        panic!()
                    };
                    // info!("\nhamming_cr {} - cr: {}\n", tags[0].0, self.m_cr);
                }
                sio.output(0).add_tag(
                    0,
                    Tag::NamedAny(
                        "frame_info".to_string(),
                        Box::new(Pmt::MapStrPmt(tags.pop_front().unwrap().1)),
                    ),
                );
            }
        }
        nitems_to_process = min(nitems_to_process, min(n_input, output.len()));
        let cr_app = if self.is_header { 4 } else { self.m_cr };
        let cw_len = cr_app + 4;
        for i in 0..nitems_to_process {
            if self.m_soft_decoding {
                let input = sio.input(0).slice::<[LLR; 8]>();
                // #ifdef GRLORA_DEBUG
                //                     // convert LLR to binary for debug
                //                     uint8_t x(0);
                //                     for (int i(0); i < cw_len; i++) x += (codeword_LLR[i] > 0) << (7 - i);
                //                     std::bitset<8> X(x);
                //                     std::cout << "Hamming in-symbol: " << +x << " " << X << std::endl;
                // #endif
                /*  Hamming Look-up Table generation, parity bits formula with data [d0 d1 d2 d3]:
                 *      p0 = d0 ^ d1 ^ d2;     ^ = xor
                 *      p1 = d1 ^ d2 ^ d3;
                 *      p2 = d0 ^ d1 ^ d3;
                 *      p3 = d0 ^ d2 ^ d3;
                 *
                 *      p = d0 ^ d1 ^ d2 ^ d3;  for CR=4/5
                 *
                 *      For LUT, store the decimal value instead of bit matrix, same LUT for CR 4/6, 4/7 and 4/8 (just crop)
                 *      e.g.    139 = [ 1 0 0 0 | 1 0 1 1 ] = [ d0 d1 d2 d3 | p0 p1 p2 p3]
                 */
                let cw_LUT: [u8; CW_NBR] = [
                    0, 23, 45, 58, 78, 89, 99, 116, 139, 156, 166, 177, 197, 210, 232, 255,
                ];
                let cw_LUT_cr5: [u8; CW_NBR] = [
                    0, 24, 40, 48, 72, 80, 96, 120, 136, 144, 160, 184, 192, 216, 232, 240,
                ]; // Different for cr = 4/5
                let mut cw_proba: [LLR; CW_NBR] = [0.; CW_NBR];
                for n in 0..CW_NBR {
                    // for all possible codeword
                    for j in 0..cw_len {
                        // for all codeword bits
                        // Select correct bit
                        let bit: bool = ((if cr_app != 1 {
                            // from correct LUT
                            cw_LUT[n]
                        } else {
                            cw_LUT_cr5[n]
                        }) >> (8 - cw_len))  // crop table (cr)
                            & (1_u8 << (cw_len - 1 - j))  // bit position mask
                            != 0;
                        // if LLR > 0 --> 1     if LLR < 0 --> 0
                        if (bit && input[i][j] > 0.) || (!bit && input[i][j] < 0.) {
                            // if correct bit 1-->1 or 0-->0
                            cw_proba[n] += input[i][j].abs();
                        } else {
                            // if incorrect bit 0-->1 or 1-->0
                            cw_proba[n] -= input[i][j].abs(); // penalty
                        } // can be optimized in 1 line: ... + ((cond)? 1 : -1) * abs(codeword_LLR[j]); but less readable
                    }
                }
                // Select the codeword with the maximum probability (ML)
                let idx_max = argmax_float(&cw_proba);
                // convert LLR to binary => Hard decision
                let data_nibble_soft: u8 = cw_LUT[idx_max] >> 4; // Take data bits of the correct codeword (=> discard hamming code part)

                // #ifdef GRLORA_DEBUG
                //                     // for (int n = 0; n < cw_nbr; n++) std::cout << cw_proba[n] << std::endl;
                //                     std::cout << "correct cw " << unsigned(correct_cw) << " with proba " << cw_proba[idx_max] << " idxm " << unsigned(idx_max) << std::endl;
                //
                //                     /*if ( std::find(cw_LUT.begin(), cw_LUT.end(), x) != cw_LUT.end() )
                //                         std::cout << "LUT " << unsigned(x) << std::endl;
                //                     else
                //                         std::cout << "NOT in LUT " << unsigned(x) << std::endl;*/
                // #endif
                // Output the most probable data nibble
                // and reversed bit order MSB<=>LSB
                output[i] = ((data_nibble_soft & 0b0001) << 3)
                    + ((data_nibble_soft & 0b0010) << 1)
                    + ((data_nibble_soft & 0b0100) >> 1)
                    + ((data_nibble_soft & 0b1000) >> 3);
            } else {
                // Hard decoding
                let input = sio.input(0).slice::<u8>();
                //                     std::vector<bool> data_nibble(4, 0);
                //                     bool s0, s1, s2 = 0;
                //                     int syndrom = 0;
                //                     std::vector<bool> codeword;
                //
                let codeword = int2bool(input[i] as u16, cr_app + 4);
                let mut data_nibble: Vec<bool> = codeword[0..4].to_vec();
                data_nibble.reverse(); // reorganized msb-first
                match cr_app {
                    3 => {
                        // get syndrom
                        let s0 = codeword[0] ^ codeword[1] ^ codeword[2] ^ codeword[4];
                        let s1 = codeword[1] ^ codeword[2] ^ codeword[3] ^ codeword[5];
                        let s2 = codeword[0] ^ codeword[1] ^ codeword[3] ^ codeword[6];
                        let syndrom = s0 as u8 + ((s1 as u8) << 1) + ((s2 as u8) << 2);

                        match syndrom {
                            5 => data_nibble[3] = !data_nibble[3],
                            7 => data_nibble[2] = !data_nibble[2],
                            3 => data_nibble[1] = !data_nibble[1],
                            6 => data_nibble[0] = !data_nibble[0],
                            _ => {} // either parity bit wrong or no error
                        }
                    }
                    // TODO noops
                    // 2 => {
                    //     s0 = codeword[0] ^ codeword[1] ^ codeword[2] ^ codeword[4];
                    //     s1 = codeword[1] ^ codeword[2] ^ codeword[3] ^ codeword[5];
                    //
                    //     if (s0 | s1) {}
                    // }
                    // case 1:
                    //     if (!(count(codeword.begin(), codeword.end(), true) % 2)) {
                    //     }
                    //     break;
                    // case 4:
                    //     if (!(count(codeword.begin(), codeword.end(), true) % 2))  // Don't correct if even number of errors
                    //         break;
                    _ => {}
                }
                output[i] = bool2int(&data_nibble) as u8;
            }
        }
        // if nitems_to_process > 0 {
        //     info!("HammingDec: producing {} samples", nitems_to_process);
        // }
        sio.input(0).consume(nitems_to_process);
        sio.output(0).produce(nitems_to_process);

        Ok(())
    }
}
