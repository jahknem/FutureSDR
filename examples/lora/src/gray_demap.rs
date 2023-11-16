use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use futuresdr::log::info;
use std::cmp::min;

use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;

use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::Tag;
use futuresdr::runtime::WorkIo;
use futuresdr::runtime::{Block, ItemTag};

use crate::utilities::*;

pub struct GrayDemap {
    m_sf: usize,
}

impl GrayDemap {
    pub fn new(sf: usize) -> Block {
        Block::new(
            BlockMetaBuilder::new("GrayDemap").build(),
            StreamIoBuilder::new()
                .add_input::<u16>("in")
                .add_output::<u16>("out")
                .build(),
            MessageIoBuilder::new().build(),
            GrayDemap { m_sf: sf },
        )
    }

    // fn set_sf(&mut self, sf: usize) {
    //     self.m_sf = sf;
    // }
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
        let input = sio.input(0).slice::<u16>();
        let out = sio.output(0).slice::<u16>();
        let nitems_to_process = min(input.len(), out.len());
        let tags: Vec<(usize, Tag)> = sio
            .input(0)
            .tags()
            .iter()
            .filter_map(|x| {
                let ItemTag { index, tag } = x;
                if *index < nitems_to_process {
                    Some((*index, tag.clone()))
                } else {
                    None
                }
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
                out[i] ^= input[i] >> j as u16;
            }
            //do the shift of 1
            out[i] = my_modulo((out[i] + 1) as isize, 1 << self.m_sf) as u16;
            // #ifdef GRLORA_DEBUG
            // std::cout<<"0x"<<out[i]<<std::dec<<std::endl;
            // #endif
        }
        info! {"GrayDemap: producing {} samples.", nitems_to_process};
        sio.input(0).consume(nitems_to_process);
        sio.output(0).produce(nitems_to_process);
        Ok(())
    }
}
