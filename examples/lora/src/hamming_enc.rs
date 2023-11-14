use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::mem;
// use futuresdr::futures::FutureExt;
use futuresdr::futures::channel::mpsc;
use futuresdr::futures::executor::block_on;
use futuresdr::futures_lite::StreamExt;
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

use rustfft::{FftDirection, FftPlanner};

pub struct HammingEnc {
    m_cr: u8,     // Transmission coding rate
    m_sf: usize,  // Transmission spreading factor
    m_cnt: usize, // count the number of processed items in the current frame
}

impl HammingEnc {
    pub fn new(cr: u8, sf: usize) -> Block {
        Block::new(
            BlockMetaBuilder::new("HammingEnc").build(),
            StreamIoBuilder::new()
                .add_input::<u8>("in")
                .add_output::<u8>("out")
                .build(),
            MessageIoBuilder::new().build(),
            HammingEnc {
                m_sf: sf,
                m_cr: cr,
                m_cnt: 0, // implicit
            },
        )
        // set_tag_propagation_policy(TPP_ONE_TO_ONE);  // TODO
    }

    fn set_cr(&mut self, cr: u8) {
        self.m_cr = cr;
    }

    fn set_sf(&mut self, sf: usize) {
        self.m_sf = sf;
    }

    fn get_cr(&self) -> usize {
        self.m_cr as usize
    }
}

#[async_trait]
impl Kernel for HammingEnc {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input = sio.input(0).slice::<u8>();
        let out = sio.output(0).slice::<u8>();
        let mut nitems_to_process = min(input.len(), out.len());
        let mut tags: Vec<(usize, Tag)> = sio
            .input(0)
            .tags()
            .iter()
            .filter_map(|x| match x {
                ItemTag { index, tag } => {
                    if *index < nitems_to_process {
                        Some((*index, tag.clone()))
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .collect();
        if tags.len() > 0 {
            if tags[0].0 != 0 {
                nitems_to_process = min(nitems_to_process, tags[0].0);
            } else {
                if tags.len() >= 2 {
                    nitems_to_process = min(nitems_to_process, tags[1].0);
                }
                self.m_cnt = 0;
            }
        }
        for (idx, tag) in tags {
            if idx < nitems_to_process {
                sio.output(0).add_tag(idx, tag)
            }
        } // propagate tags
        for i in 0..nitems_to_process {
            // #ifdef GRLORA_DEBUG
            //         std::cout << std::hex << (int)in_data[i] << "   ";
            // #endif
            let cr_app = if self.m_cnt < self.m_sf - 2 {
                4
            } else {
                self.m_cr
            };
            let data_bin = int2bool(input[i] as u16, 4);
            //the data_bin is msb first
            if cr_app != 1 {
                //need hamming parity bits
                let p0 = (data_bin[3] ^ data_bin[2] ^ data_bin[1]) as u8;
                let p1 = (data_bin[2] ^ data_bin[1] ^ data_bin[0]) as u8;
                let p2 = (data_bin[3] ^ data_bin[2] ^ data_bin[0]) as u8;
                let p3 = (data_bin[3] ^ data_bin[1] ^ data_bin[0]) as u8;
                //we put the data LSB first and append the parity bits
                out[i] = ((data_bin[3] as u8) << 7
                    | (data_bin[2] as u8) << 6
                    | (data_bin[1] as u8) << 5
                    | (data_bin[0] as u8) << 4
                    | p0 << 3
                    | p1 << 2
                    | p2 << 1
                    | p3)
                    >> (4 - cr_app);
            } else {
                // coding rate = 4/5 we add a parity bit
                let p4 = (data_bin[0] ^ data_bin[1] ^ data_bin[2] ^ data_bin[3]) as u8;
                out[i] = (data_bin[3] as u8) << 4
                    | (data_bin[2] as u8) << 3
                    | (data_bin[1] as u8) << 2
                    | (data_bin[0] as u8) << 1
                    | p4;
            }
            // #ifdef GRLORA_DEBUG
            //         std::cout << std::hex << (int)out[i] << std::dec << std::endl;
            // #endif
            self.m_cnt += 1;
        }
        sio.input(0).consume(nitems_to_process);
        sio.output(0).produce(nitems_to_process);
        Ok(())
    }
}
