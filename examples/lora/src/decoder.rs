use futuresdr::anyhow::Result;
use futuresdr::log::info;
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

    fn crc16(data: &[u8]) -> u16 {
        let mut crc: u16 = 0x0000;
        for byte in data.iter() {
            let mut new_byte = *byte;
            for _ in 0..8 {
                if ((crc & 0x8000) >> 8) as u8 ^ (new_byte & 0x80) != 0 {
                    crc = (crc << 1) ^ 0x1021;
                } else {
                    crc <<= 1;
                }
                new_byte <<= 1;
            }
        }
        crc
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
                    let mut dewhitened: Vec<u8> = vec![];
                    let start = if frame.implicit_header { 0 } else { 5 };
                    let end = if frame.has_crc {
                        frame.nibbles.len() - 4
                    } else {
                        frame.nibbles.len()
                    };

                    let slice = &frame.nibbles[start..end];

                    for (i, c) in slice.chunks_exact(2).enumerate() {
                        let low_nib = c[0] ^ (WHITENING_SEQ[i] & 0x0F);
                        let high_nib = c[1] ^ ((WHITENING_SEQ[i] & 0xF0) >> 4);
                        dewhitened.push((high_nib << 4) | low_nib);
                    }

                    //info!("..:: Payload");

                    if frame.has_crc {
                        let l = frame.nibbles.len();
                        let low_nib = frame.nibbles[l - 4];
                        let high_nib = frame.nibbles[l - 3];
                        dewhitened.push((high_nib << 4) | low_nib);
                        let low_nib = frame.nibbles[l - 2];
                        let high_nib = frame.nibbles[l - 1];
                        dewhitened.push((high_nib << 4) | low_nib);

                        let l = dewhitened.len();
                        let mut crc = Self::crc16(&dewhitened[0..l - 4]);
                        // XOR the obtained CRC with the last 2 data bytes
                        crc = crc ^ dewhitened[l - 3] as u16 ^ ((dewhitened[l - 4] as u16) << 8);
                        let crc_valid: bool =
                            ((dewhitened[l - 2] as u16) + ((dewhitened[l - 1] as u16) << 8)) as i32
                                == crc as i32;
                        if !crc_valid {
                            //info!("crc check failed");
                            return Ok(Pmt::Ok);
                        } else {
                            //info!("crc check passed");
                        }
                    }
                    let data = String::from_utf8_lossy(&dewhitened);
                    println!("received: {}", data);

                    let mut rftap = vec![0; dewhitened.len() + 12 + 15];
                    rftap[0..4].copy_from_slice("RFta".as_bytes());
                    rftap[4..6].copy_from_slice(&3u16.to_le_bytes());
                    rftap[6..8].copy_from_slice(&1u16.to_le_bytes());
                    rftap[8..12].copy_from_slice(&270u32.to_le_bytes());
                    rftap[12] = 0; // version
                    rftap[13] = 0; // padding
                    rftap[14..16].copy_from_slice(&15u16.to_be_bytes()); // header len
                    rftap[16..20].copy_from_slice(&868100000u32.to_be_bytes()); // frequency
                    rftap[20] = 1; // bandwidth
                    rftap[21] = 7; // spreading factor
                    rftap[22] = 0; // packet rssi
                    rftap[23] = 0; // max_rssi
                    rftap[24] = 0; // current_rssi
                    rftap[25] = 0; // snr
                    rftap[26] = 0x12; // sync word
                    rftap[27..].copy_from_slice(&dewhitened);
                    mio.output_mut(1).post(Pmt::Blob(rftap.clone())).await;

                    let data = String::from_utf8_lossy(&dewhitened);
                    //info!("received frame: {}", data);
                    mio.output_mut(0).post(Pmt::Blob(dewhitened)).await;

                    Pmt::Ok
                } else {
                    Pmt::InvalidValue
                }
            }
            _ => Pmt::InvalidValue,
        };
        Ok(ret)
    }
}

#[async_trait]
impl Kernel for Decoder {}
