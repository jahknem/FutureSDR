use futuresdr::anyhow::Result;
use futuresdr::macros::async_trait;
use futuresdr::macros::message_handler;
use futuresdr::runtime::Block;
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::WorkIo;

use crate::utilities::*;
use crate::Frame;

pub struct Decoder;

impl Decoder {
    pub fn new() -> Block {
        Block::new(
            BlockMetaBuilder::new("Decoder").build(),
            StreamIoBuilder::new().build(),
            MessageIoBuilder::new()
                .add_input("in", Self::handler)
                .add_output("data")
                .add_output("rftap")
                .build(),
            Decoder,
        )
    }

    #[message_handler]
    async fn handler(
        &mut self,
        _io: &mut WorkIo,
        mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
        pmt: Pmt,
    ) -> Result<Pmt> {
        let ret = match pmt {
            Pmt::Any(a) => {
                if let Some(frame) = a.downcast_ref::<Frame>() {
                    dbg!(&frame);
                    let mut dewhitened: Vec<u8> = vec![];
                    let start = if frame.implicit_header { 0 } else { 5 };
                    let end = if frame.has_crc {
                        frame.nibbles.len() - 2
                    } else {
                        frame.nibbles.len()
                    };

                    let slice = &frame.nibbles[start..end];

                    for (i, c) in slice.chunks_exact(2).enumerate() {
                        let low_nib = c[0] ^ (WHITENING_SEQ[i] & 0x0F);
                        let high_nib = c[1] ^ ((WHITENING_SEQ[i] & 0xF0) >> 4);
                        dewhitened.push((high_nib << 4) | low_nib);
                    }

                    if frame.has_crc {
                        let l = frame.nibbles.len();
                        let low_nib = frame.nibbles[l - 4];
                        let high_nib = frame.nibbles[l - 3];
                        dewhitened.push((high_nib << 4) | low_nib);
                        let low_nib = frame.nibbles[l - 2];
                        let high_nib = frame.nibbles[l - 1];
                        dewhitened.push((high_nib << 4) | low_nib);
                    }

                    if !frame.implicit_header {
                        let mut rftap = vec![0; dewhitened.len() + 12 + 5];
                        rftap[0..4].copy_from_slice("RFta".as_bytes());
                        rftap[4..6].copy_from_slice(&3u16.to_le_bytes());
                        rftap[6..8].copy_from_slice(&1u16.to_le_bytes());
                        rftap[8..12].copy_from_slice(&105u32.to_le_bytes());
                        rftap[12..17].copy_from_slice(&frame.nibbles[0..5]);
                        rftap[17..].copy_from_slice(&dewhitened);
                        mio.output_mut(1).post(Pmt::Blob(rftap.clone())).await;
                    }

                    mio.output_mut(0).post(Pmt::Blob(dewhitened)).await;

                    Pmt::Ok
                } else {
                    Pmt::InvalidValue
                }
            },
            _ => Pmt::InvalidValue,
        };
        Ok(ret)
    }
}

#[async_trait]
impl Kernel for Decoder {}
