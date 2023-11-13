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

pub struct GrayDemap {
    m_sf: usize,
}

impl GrayDemap {
    pub fn new(sf: usize) -> Block {
        Block::new(
            BlockMetaBuilder::new("GrayDemap").build(),
            StreamIoBuilder::new()
                .add_input::<usize>("in")
                .add_output::<usize>("out")
                .build(),
            MessageIoBuilder::new().build(),
            GrayDemap { m_sf: sf },
        )
        // set_tag_propagation_policy(TPP_ONE_TO_ONE);  // TODO
    }

    fn set_sf(&mut self, sf: usize) {
        self.m_sf = sf;
    }
}

#[async_trait]
impl Kernel for GrayDemap {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input = sio.input(0).slice::<usize>();
        let out = sio.output(0).slice::<usize>();
        let nitems_to_process = min(input.len(), out.len());
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
        for (idx, tag) in tags {
            sio.output(0).add_tag(idx, tag)
        }
        for i in 0..nitems_to_process {
            // #ifdef GRLORA_DEBUG
            // std::cout<<std::hex<<"0x"<<in[i]<<" -->  ";
            // #endif
            out[i] = input[i];
            for j in 1..self.m_sf {
                out[i] = out[i] ^ (input[i] >> j);
            }
            //do the shift of 1
            out[i] = my_modulo((out[i] + 1) as isize, (1 << self.m_sf));
            // #ifdef GRLORA_DEBUG
            // std::cout<<"0x"<<out[i]<<std::dec<<std::endl;
            // #endif
        }
        sio.input(0).consume(nitems_to_process);
        sio.output(0).produce(nitems_to_process);
        Ok(())
    }
}
