use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;

use std::collections::HashMap;

// use futuresdr::futures::FutureExt;
use futuresdr::log::{info, warn};

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
use std::string::String;

pub struct CrcVerif {
    m_payload_len: usize, // Payload length in bytes
    m_crc_presence: bool, // Indicate if there is a payload CRC
    // m_crc: u16,                        // The CRC calculated from the received payload
    // message_str: String,               // The payload string
    // m_char: char,                     // A new char of the payload
    // new_frame: bool,                  //indicate a new frame
    in_buff: Vec<u8>,       // input buffer containing the data bytes and CRC if any
    print_rx_msg: bool,     // print received message in terminal or not
    output_crc_check: bool, // output the result of the payload CRC check
    curent_tag: HashMap<String, Pmt>, // the most recent tag for the packet we are currently processing

    cnt: usize, // count the number of frame
}

impl CrcVerif {
    pub fn new(print_rx_msg: bool, output_crc_check: bool) -> Block {
        Block::new(
            BlockMetaBuilder::new("CrcVerif").build(),
            StreamIoBuilder::new()
                .add_input::<u8>("in")
                .add_output::<u8>("out")
                .add_output::<bool>("out1")
                .build(),
            MessageIoBuilder::new().add_output("msg").build(),
            CrcVerif {
                m_payload_len: 0,
                m_crc_presence: false,
                in_buff: vec![],
                print_rx_msg,
                output_crc_check,
                curent_tag: HashMap::new(),
                cnt: 0,
            },
        )
        // set_tag_propagation_policy(TPP_DONT); // TODO
    }

    /**
     *  \brief  Calculate the CRC 16 using poly=0x1021 and Init=0x0000
     *
     *  \param  data
     *          The pointer to the data beginning.
     *  \param  len
     *          The length of the data in bytes.
     */
    fn crc16(data: &[u8], len: usize) -> u16 {
        let mut crc: u16 = 0x0000;
        for byte in data[0..len].iter() {
            let mut new_byte = *byte;
            for _j in 0..8 {
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
}

#[async_trait]
impl Kernel for CrcVerif {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        mio: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input = sio.input(0).slice::<u8>();
        let nitem_to_consume = input.len();
        let _nitem_to_produce_0: usize = 0;

        let out = if !sio.outputs().is_empty() {
            Some(sio.output(0).slice::<u8>())
        } else {
            None
        };
        let out_crc = if self.output_crc_check {
            Some(sio.output(0).slice::<bool>())
        } else {
            None
        };

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
            self.m_crc_presence = if let Pmt::Bool(tmp) = tag.get("crc").unwrap() {
                *tmp
            } else {
                panic!()
            };
            self.m_payload_len = if let Pmt::Usize(tmp) = tag.get("pay_len").unwrap() {
                *tmp
            } else {
                panic!()
            };
            self.curent_tag = tag;
            // info!("{} {}", self.m_payload_len, nitem_to_process);
            // info!("crc_crc {} - crc: {} - pay_len: {}", tags[0].offset, self.m_crc_presence, self.m_payload_len);
        }
        // greedily consume input and buffer until enough available
        self.in_buff
            .append(&mut input[0..nitem_to_consume].to_vec());
        sio.input(0).consume(nitem_to_consume);
        // process buffered samples
        if let Some(ref out_buf) = out {
            if out_buf.len() <= self.m_payload_len {
                warn!("not enough space in out buffer, waiting for more space.");
                return Ok(());
            }
        }
        if let Some(ref out_crc_buf) = out_crc {
            if out_crc_buf.is_empty() {
                warn!("not enough space in crc out buffer, waiting for more space.");
                return Ok(());
            }
        }
        if self.in_buff.len() >= self.m_payload_len + if self.m_crc_presence { 2 } else { 0 } {
            if self.m_crc_presence {
                // wait for all the payload to come
                if self.m_payload_len < 2 {
                    // undefined CRC
                    warn!("CRC not supported for payload smaller than 2 bytes");
                    return Ok(());
                }
                // calculate CRC on the N-2 firsts data bytes
                let mut m_crc = CrcVerif::crc16(&(self.in_buff), self.m_payload_len - 2);
                // XOR the obtained CRC with the last 2 data bytes
                m_crc = m_crc
                    ^ self.in_buff[self.m_payload_len - 1] as u16
                    ^ ((self.in_buff[self.m_payload_len - 2] as u16) << 8);
                // # ifdef
                // GRLORA_DEBUG
                // for (int i = 0; i < (int)m_payload_len + 2; i+ +)
                // std::cout << std::hex << (int)
                // in_buff[i] << std::dec << std::endl;
                // std::cout << "Calculated " << std::hex << m_crc << std::dec << std::endl;
                // std::cout << "Got " << std::hex << (in_buff[m_payload_len] + (in_buff[m_payload_len + 1] << 8)) << std::dec << std::endl;
                // # endif
                let crc_valid: bool = (self.in_buff[self.m_payload_len] as u16
                    + ((self.in_buff[self.m_payload_len + 1] as u16) << 8))
                    as i32
                    - m_crc as i32
                    == 0;
                self.curent_tag
                    .insert("crc_valid".to_string(), Pmt::Bool(crc_valid));
                if let Some(out_crc_buf) = out_crc {
                    out_crc_buf[0] = crc_valid;
                    sio.output(1).produce(1);
                }
                if self.print_rx_msg {
                    if crc_valid {
                        info!("CRC valid!");
                    } else {
                        info!("CRC invalid!");
                    }
                }
            }

            // get payload as string
            let message_str = self.in_buff[0..self.m_payload_len]
                .iter()
                .map(|x| *x as char)
                .collect::<String>();
            if let Some(out_buf) = out {
                out_buf[0..self.m_payload_len]
                    .copy_from_slice(&self.in_buff[0..self.m_payload_len]);
                let new_tag = self.curent_tag.clone();
                sio.output(0).add_tag(
                    0,
                    Tag::NamedAny("frame_info".to_string(), Box::new(Pmt::MapStrPmt(new_tag))),
                );
            }
            self.cnt += 1;
            if self.print_rx_msg {
                info!("rx msg: {}", message_str);
            }
            mio.output_mut(0).post(Pmt::String(message_str)).await;
            self.in_buff
                .drain(0..(self.m_payload_len + if self.m_crc_presence { 2 } else { 0 }));
            sio.output(0).produce(self.m_payload_len);
        }

        Ok(())
    }
}
