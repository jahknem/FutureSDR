use futuresdr::anyhow::Result;
use futuresdr::macros::async_trait;
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
use std::cmp::min;
use std::collections::HashMap;
use std::collections::VecDeque;

use crate::utilities::*;

pub struct GrayMapping {
    m_soft_decoding: bool, // Hard/Soft decoding
}

impl GrayMapping {
    pub fn new(soft_decoding: bool) -> Block {
        let mut sio = StreamIoBuilder::new();
        if soft_decoding {
            sio = sio.add_input::<[LLR; MAX_SF]>("in");
            sio = sio.add_output::<[LLR; MAX_SF]>("out");
        } else {
            sio = sio.add_input::<u16>("in");
            sio = sio.add_output::<u16>("out");
        }
        Block::new(
            BlockMetaBuilder::new("GrayMapping").build(),
            sio.build(),
            MessageIoBuilder::new().build(),
            GrayMapping {
                m_soft_decoding: soft_decoding,
            },
        )
    }
}

#[async_trait]
impl Kernel for GrayMapping {
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
            sio.output(0).slice::<[LLR; MAX_SF]>().len()
        } else {
            sio.output(0).slice::<u16>().len()
        };
        let mut nitems_to_process = min(n_input, n_output);
        if nitems_to_process == 0 {
            return Ok(());
        }
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
        //             get_tags_in_window(tags, 0, 0, ninput_items[0], pmt::string_to_symbol("new_frame"));
        if !tags.is_empty() {
            if tags[0].0 != 0 {
                nitems_to_process = tags[0].0; // only use symbol until the next frame begin (SF might change)
            } else {
                if tags.len() >= 2 {
                    nitems_to_process = tags[1].0; //  - tags[0].0; (== 0)
                }

                // pmt::pmt_t err = pmt::string_to_symbol("error");
                // bool is_header = pmt::to_bool(pmt::dict_ref(tags[0].value, pmt::string_to_symbol("is_header"), err));
                // if (is_header) // new frame beginning
                // {
                // int sf = pmt::to_long(pmt::dict_ref(tags[0].value, pmt::string_to_symbol("sf"), err));
                // m_sf = sf;  // TODO noop
                // }
                sio.output(0).add_tag(
                    0,
                    Tag::NamedAny(
                        "frame_info".to_string(),
                        Box::new(Pmt::MapStrPmt(tags.pop_front().unwrap().1)),
                    ),
                );
            }
        }

        if self.m_soft_decoding {
            let input = sio.input(0).slice::<[LLR; MAX_SF]>();
            let output = sio.output(0).slice::<[LLR; MAX_SF]>();
            // No gray mapping , it has as been done directly in fft_demod block => block "bypass"
            output[0..nitems_to_process].copy_from_slice(&input[0..nitems_to_process]);
        } else {
            let input = sio.input(0).slice::<u16>();
            let output = sio.output(0).slice::<u16>();
            output[0..nitems_to_process].copy_from_slice(
                &input[0..nitems_to_process]
                    .iter()
                    .map(|x| *x ^ (*x >> 1))
                    .collect::<Vec<u16>>(), // Gray Demap
            );
        }

        // #ifdef GRLORA_DEBUG
        //                 std::cout << std::hex << "0x" << in[i] << " ---> "
        //                           << "0x" << out[i] << std::dec << std::endl;
        // #endif

        // if nitems_to_process > 0 {
        //     info!("GrayMapping: producing {} samples", nitems_to_process);
        // }
        sio.input(0).consume(nitems_to_process);
        sio.output(0).produce(nitems_to_process);
        Ok(())
    }
}
