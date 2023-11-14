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

pub struct AddCrc {
    m_has_crc: bool,      //indicate the presence of a payload CRC
    m_payload: Vec<char>, // payload data
    m_payload_len: usize, // length of the payload in Bytes
    m_frame_len: usize,   // length of the frame in number of gnuradio items
    m_cnt: usize,         // counter of the number of symbol in frame
}

impl AddCrc {
    pub fn new(has_crc: bool) -> Block {
        Block::new(
            BlockMetaBuilder::new("AddCrc").build(),
            StreamIoBuilder::new()
                .add_input::<u8>("in")
                .add_output::<u8>("out")
                .build(),
            MessageIoBuilder::new().build(),
            AddCrc {
                m_has_crc: has_crc,
                m_payload: vec![],
                m_payload_len: 0, // implicit
                m_frame_len: 0,
                m_cnt: 0,
            },
        )
    }

    fn crc16(crcValue_in: u16, newByte_tmp: u8) -> u16 {
        let mut crcValue = crcValue_in;
        let mut newByte = newByte_tmp as u16;
        for i in 0..8 {
            if ((crcValue & 0x8000) >> 8) ^ (newByte & 0x80) != 0 {
                crcValue = (crcValue << 1) ^ 0x1021;
            } else {
                crcValue = crcValue << 1;
            }
            newByte <<= 1;
        }
        return crcValue;
    }
}

#[async_trait]
impl Kernel for AddCrc {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input = sio.input(0).slice::<u8>();
        let out = sio.output(0).slice::<u8>();
        let mut nitems_to_produce = 0;
        let noutput_items = max(0, out.len() - 4);
        let mut nitems_to_process = min(input.len(), noutput_items);
        let tags: Vec<(usize, String)> = sio
            .input(0)
            .tags()
            .iter()
            .filter_map(|x| match x {
                ItemTag {
                    index,
                    tag: Tag::NamedAny(n, val),
                } => {
                    if n == "payload_str" {
                        match (**val).downcast_ref().unwrap() {
                            Pmt::String(payload) => Some((*index, payload.clone())),
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
                nitems_to_process = min(tags[0].0, noutput_items);
            } else {
                if tags.len() >= 2 {
                    nitems_to_process = min(tags[1].0, noutput_items);
                }
                self.m_payload = tags[0].1.chars().collect();
                //pass tags downstream
                if nitems_to_process > 0 {
                    let tags_tmp: Vec<(usize, usize)> = sio
                        .input(0)
                        .tags()
                        .iter()
                        .filter_map(|x| match x {
                            ItemTag {
                                index,
                                tag: Tag::NamedAny(n, val),
                            } => {
                                if n == "frame_len" {
                                    match (**val).downcast_ref().unwrap() {
                                        Pmt::Usize(frame_len) => Some((*index, *frame_len)),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        })
                        .collect();
                    if tags_tmp.len() > 0 {
                        self.m_frame_len = tags_tmp[0].1;
                        sio.output(0).add_tag(
                            0,
                            Tag::NamedAny(
                                "frame_len".to_string(),
                                Box::new(Pmt::Usize(
                                    self.m_frame_len + if self.m_has_crc { 4 } else { 0 },
                                )),
                            ),
                        );
                    }
                }
                self.m_cnt = 0;
            }
        }
        if nitems_to_process == 0 {
            if out.len() == 0 {
                warn!("AddCrc: no space in output buffer, waiting for more.");
            } else {
                warn!("AddCrc: no samples in input buffer, waiting for more.");
            }
            return Ok(());
        }
        self.m_cnt += nitems_to_process;
        if self.m_has_crc && self.m_cnt == self.m_frame_len {
            //append the CRC to the payload
            let mut crc: u16 = 0x0000;
            self.m_payload_len = self.m_payload.len();
            //calculate CRC on the N-2 firsts data bytes using Poly=1021 Init=0000
            for i in 0..(self.m_payload_len - 2) {
                crc = Self::crc16(crc, self.m_payload[i] as u8);
            }
            //XOR the obtained CRC with the last 2 data bytes
            crc = crc
                ^ (self.m_payload[self.m_payload_len - 1] as u16)
                ^ ((self.m_payload[self.m_payload_len - 2] as u16) << 8);
            //Place the CRC in the correct output nibble
            out[nitems_to_process] = (crc & 0x000F) as u8;
            out[nitems_to_process + 1] = ((crc & 0x00F0) >> 4) as u8;
            out[nitems_to_process + 2] = ((crc & 0x0F00) >> 8) as u8;
            out[nitems_to_process + 3] = ((crc & 0xF000) >> 12) as u8;
            nitems_to_produce = nitems_to_process + 4;
            self.m_payload = vec![];
        } else {
            nitems_to_produce = nitems_to_process;
        }
        out[0..nitems_to_process].copy_from_slice(&input[0..nitems_to_process]);
        sio.input(0).consume(nitems_to_process);
        sio.output(0).produce(nitems_to_produce);
        Ok(())
    }
}
