use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use std::cmp::min;
use std::collections::HashMap;
use std::f32::consts::PI;
use std::mem;
// use futuresdr::futures::FutureExt;
use futuresdr::log::warn;
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

pub struct Deinterleaver {
    m_sf: usize, // Transmission Spreading factor
    m_cr: usize, // Transmission Coding rate
    // sf_app: usize,         // Spreading factor to use to deinterleave
    // cw_len: usize,         // Length of a codeword
    m_is_header: bool, // Indicate that we need to deinterleave the first block with the default header parameters (cr=4/8, reduced rate)
    m_soft_decoding: bool, // Hard/Soft decoding
    m_ldro: bool,      // use low datarate optimization mode
}

impl Deinterleaver {
    pub fn new(soft_decoding: bool) -> Block {
        let mut sio = StreamIoBuilder::new();
        if soft_decoding {
            sio = sio.add_input::<[LLR; MAX_SF]>("in");
            sio = sio.add_output::<[LLR; 8]>("out");
        } else {
            sio = sio.add_input::<u16>("in");
            sio = sio.add_output::<u8>("out");
        }
        Block::new(
            BlockMetaBuilder::new("Deinterleaver").build(),
            sio.build(),
            MessageIoBuilder::new().build(),
            Deinterleaver {
                // m_sf: sf,
                m_soft_decoding: soft_decoding,
                m_sf: 0,
                m_cr: 0,
                m_is_header: false,
                m_ldro: false,
            },
        )
        // set_tag_propagation_policy(TPP_DONT); // TODO
    }

    // void deinterleaver_impl::forecast(int noutput_items, gr_vector_int &ninput_items_required) {
    //       ninput_items_required[0] = 4;
    //   }
}

#[async_trait]
impl Kernel for Deinterleaver {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let n_input = if self.m_soft_decoding {
            sio.input(0).slice::<[LLR; MAX_SF]>().len()
        } else {
            sio.input(0).slice::<u16>().len()
        };
        let n_output = if self.m_soft_decoding {
            sio.output(0).slice::<[LLR; 8]>().len()
        } else {
            sio.output(0).slice::<u8>().len()
        };
        // let mut nitems_to_process = min(n_input, n_output);

        // const uint16_t *in1 = (const uint16_t *)input_items[0];
        // const LLR *in2 = (const LLR *)input_items[0];
        // uint8_t *out1 = (uint8_t *)output_items[0];
        // LLR *out2 = (LLR *)output_items[0];

        let tag_tmp: Option<HashMap<String, Pmt>> =
            sio.input(0).tags().iter().find_map(|x| match x {
                ItemTag {
                    index: _,
                    tag: Tag::NamedAny(n, val),
                } => {
                    if n == "frame_info" {
                        match (**val).downcast_ref().unwrap() {
                            Pmt::MapStrPmt(map) => Some(map.clone()),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            });
        if let Some(tag) = tag_tmp {
            self.m_is_header = if let Pmt::Bool(tmp) = tag.get("is_header").unwrap() {
                *tmp
            } else {
                panic!()
            };

            if self.m_is_header {
                self.m_sf = if let Pmt::Usize(tmp) = tag.get("sf").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                // std::cout<<"deinterleaver_header "<<tags[0].offset<<std::endl;
                // is_first = true;
            } else {
                // is_first=false;
                self.m_cr = if let Pmt::Usize(tmp) = tag.get("cr").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                self.m_ldro = if let Pmt::Bool(tmp) = tag.get("ldro").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                // std::cout<<"\ndeinter_cr "<<tags[0].offset<<" - cr: "<<(int)m_cr<<"\n";
            }
            sio.output(0).add_tag(
                0,
                Tag::NamedAny("frame_info".to_string(), Box::new(Pmt::MapStrPmt(tag))),
            );
        }
        let sf_app = if self.m_is_header || self.m_ldro {
            self.m_sf - 2
        } else {
            self.m_sf
        }; // Use reduced rate for the first block
        if n_output < sf_app {
            warn!(
                "[deinterleaver.cc] Not enough output space! {}/{}",
                n_output, sf_app
            );
            return Ok(());
        }
        let cw_len = if self.m_is_header { 8 } else { self.m_cr + 4 };
        // std::cout << "sf_app " << +sf_app << " cw_len " << +cw_len << std::endl;

        if n_input >= cw_len {
            // wait for a full block to deinterleave
            if self.m_soft_decoding {
                let input = sio.input(0).slice::<[LLR; MAX_SF]>();
                let output = sio.output(0).slice::<[LLR; 8]>();
                let mut inter_bin: Vec<[LLR; MAX_SF]> = vec![[0.; MAX_SF]; cw_len];
                let mut deinter_bin: Vec<[LLR; 8]> = vec![[0.; 8]; sf_app];
                for i in 0..cw_len {
                    // take only sf_app bits over the sf bits available
                    let input_offset = self.m_sf - sf_app;
                    let count = sf_app;
                    inter_bin[i][0..count]
                        .copy_from_slice(&input[i][input_offset..(input_offset + count)]);
                }
                // Do the actual deinterleaving
                for i in 0..cw_len {
                    for j in 0..sf_app {
                        // std::cout << "T["<<i<<"]["<<j<<"] "<< (inter_bin[i][j] > 0) << " ";
                        deinter_bin[(i - j - 1) % sf_app][i] = inter_bin[i][j];
                    }
                    // std::cout << std::endl;
                }
                for i in 0..sf_app {
                    output[i] = deinter_bin[i];
                    // Write only the cw_len bits over the 8 bits space available
                }
            } else {
                // Hard-Decoding
                //                     // Create the empty matrices
                //                     std::vector<std::vector<bool>> inter_bin(cw_len);
                //                     std::vector<bool> init_bit(cw_len, 0);
                //                     std::vector<std::vector<bool>> deinter_bin(sf_app, init_bit);
                //
                let input = sio.input(0).slice::<u16>();
                let output = sio.output(0).slice::<u8>();
                let mut inter_bin: Vec<Vec<bool>> = vec![vec![false; sf_app]; cw_len];
                let mut deinter_bin: Vec<Vec<bool>> = vec![vec![false; cw_len]; sf_app];
                // convert decimal vector to binary vector of vector
                for i in 0..cw_len {
                    inter_bin[i] = int2bool(input[i], sf_app);
                }
                // #ifdef GRLORA_DEBUG
                //                     std::cout << "interleaved----" << std::endl;
                //                     for (uint32_t i = 0u; i < cw_len; i++) {
                //                         for (int j = 0; j < int(sf_app); j++) {
                //                             std::cout << inter_bin[i][j];
                //                         }
                //                         std::cout << " " << (int)in1[i] << std::endl;
                //                     }
                //                     std::cout << std::endl;
                // #endif
                // Do the actual deinterleaving
                for i in 0..cw_len {
                    for j in 0..sf_app {
                        // std::cout << "T["<<i<<"]["<<j<<"] "<< inter_bin[i][j] << " ";
                        deinter_bin[(i - j - 1) % sf_app][i] = inter_bin[i][j];
                    }
                    // std::cout << std::endl;
                }
                // transform codewords from binary vector to dec
                for i in 0..sf_app {
                    output[i] = bool2int(&deinter_bin[i]);
                }
                // #ifdef GRLORA_DEBUG
                //                     std::cout << "codewords----" << std::endl;
                //                     for (uint32_t i = 0u; i < sf_app; i++) {
                //                         for (int j = 0; j < int(cw_len); j++) {
                //                             std::cout << deinter_bin[i][j];
                //                         }
                //                         std::cout << " 0x" << std::hex << (int)out1[i] << std::dec << std::endl;
                //                     }
                //                     std::cout << std::endl;
                // #endif
                // if(is_first)
                //     add_item_tag(0, nitems_written(0), pmt::string_to_symbol("header_len"), pmt::mp((long)sf_app));//sf_app is the header part size
            }
            sio.input(0).consume(cw_len);
            sio.output(0).produce(sf_app);
        }
        Ok(())
    }
}
