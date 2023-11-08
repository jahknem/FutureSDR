use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use std::cmp::min;
use std::collections::HashMap;
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

const HEADER_LEN: usize = 5; // size of the header in nibbles

pub struct Dewhitening {
    m_payload_len: usize, // Payload length in bytes
    m_crc_presence: bool, // indicate the precence of a CRC
    offset: usize,        // The offset in the whitening table
                          // dewhitened: Vec<u8>,  // The dewhitened bytes
}

impl Dewhitening {
    pub fn new() -> Block {
        Block::new(
            BlockMetaBuilder::new("Dewhitening").build(),
            StreamIoBuilder::new()
                .add_input::<u8>("in")
                .add_output::<u8>("out")
                .build(),
            MessageIoBuilder::new().build(),
            Dewhitening {
                m_payload_len: 0,
                m_crc_presence: false,
                offset: 0,
                // dewhitened: vec![],
            },
        )
        // set_tag_propagation_policy(TPP_DONT); // TODO
    }

    // void dewhitening_impl::header_pay_len_handler(pmt::pmt_t payload_len)
    // {
    //     m_payload_len = pmt::to_long(payload_len);
    // };
    //
    // void dewhitening_impl::new_frame_handler(pmt::pmt_t id)
    // {
    //     offset = 0;
    // }
    // void dewhitening_impl::header_crc_handler(pmt::pmt_t crc_presence)
    // {
    //     m_crc_presence = pmt::to_long(crc_presence);
    // };
}

#[async_trait]
impl Kernel for Dewhitening {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        mio: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input = sio.input(0).slice::<u8>();
        let mut out = sio.output(0).slice::<u8>();
        let mut nitem_to_process = input.len();

        let tags: Vec<(usize, HashMap<String, Pmt>)> = sio
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
                nitem_to_process = tags[0].0;
            } else {
                if tags.len() >= 2 {
                    nitem_to_process = tags[1].0 - tags[0].0;
                }

                self.m_crc_presence = if let Pmt::Bool(tmp) = tags[0].1.get("crc").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                self.m_payload_len = if let Pmt::Usize(tmp) = tags[0].1.get("pay_len").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                self.offset = 0;
                sio.output(0).add_tag(
                    0,
                    Tag::NamedAny(
                        "frame_info".to_string(),
                        Box::new(Pmt::MapStrPmt(tags[0].1.clone())),
                    ),
                );
                // std::cout<<"\ndewi_crc "<<tags[0].offset<<" - crc: "<<(int)m_crc_presence<<" - pay_len: "<<(int)m_payload_len<<"\n";
            }
        }
        let nitem_to_process = min(nitem_to_process, min(input.len(), out.len() * 2));
        let mut dewhitened: Vec<u8> = vec![];
        for i in 0..nitem_to_process / 2 {
            if self.offset < self.m_payload_len {
                let low_nib = input[2 * i] ^ (WHITENING_SEQ[self.offset] & 0x0F);
                let high_nib = input[2 * i + 1] ^ (WHITENING_SEQ[self.offset] & 0xF0) >> 4;
                dewhitened.push(high_nib << 4 | low_nib);
            } else if (self.offset < self.m_payload_len + 2) && self.m_crc_presence {
                //do not dewhiten the CRC
                let low_nib = input[2 * i];
                let high_nib = input[2 * i + 1];
                dewhitened.push(high_nib << 4 | low_nib);
            } else {
                // full packet received
                break;
            }
            self.offset += 1;
        }
        // #ifdef GRLORA_DEBUG
        //             for (unsigned int i = 0; i < dewhitened.size(); i++)
        //             {
        //                 std::cout << (char)(int)dewhitened[i] << "    0x" << std::hex << (int)dewhitened[i] << std::dec << std::endl;
        //             }
        // #endif
        out[0..dewhitened.len()].copy_from_slice(&dewhitened);
        sio.input(0).consume(dewhitened.len() * 2); //ninput_items[0]/2*2
        sio.output(0).produce(dewhitened.len());

        Ok(())
    }
}
