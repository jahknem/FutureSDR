use futuresdr::anyhow::Result;
use futuresdr::macros::async_trait;
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
use std::collections::HashMap;
use std::string::String;

pub struct CrcVerif {
    m_payload_len: usize,             // Payload length in bytes
    m_crc_presence: bool,             // Indicate if there is a payload CRC
    in_buff: Vec<u8>,                 // input buffer containing the data bytes and CRC if any
    print_rx_msg: bool,               // print received message in terminal or not
    curent_tag: HashMap<String, Pmt>, // the most recent tag for the packet we are currently processing
}

impl CrcVerif {
    pub fn new(print_rx_msg: bool) -> Block {
        Block::new(
            BlockMetaBuilder::new("CrcVerif").build(),
            StreamIoBuilder::new().add_input::<u8>("in").build(),
            MessageIoBuilder::new().add_output("msg").build(),
            CrcVerif {
                m_payload_len: 0,
                m_crc_presence: false,
                in_buff: vec![],
                print_rx_msg,
                curent_tag: HashMap::new(),
            },
        )
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
                let crc_valid: bool = (self.in_buff[self.m_payload_len] as u16
                    + ((self.in_buff[self.m_payload_len + 1] as u16) << 8))
                    as i32
                    - m_crc as i32
                    == 0;
                self.curent_tag
                    .insert("crc_valid".to_string(), Pmt::Bool(crc_valid));
                if self.print_rx_msg {
                    if crc_valid {
                        info!("CRC valid!");
                    } else {
                        info!("CRC invalid!");
                    }
                }
            }

            // get payload as string
            let blob = Pmt::Blob(Vec::from(&self.in_buff[0..self.m_payload_len]));
            let message_str = String::from_utf8_lossy(&self.in_buff[0..self.m_payload_len]);
            if self.print_rx_msg {
                info!("rx msg: {}", message_str);
            }
            mio.output_mut(0).post(blob).await;
            self.in_buff.drain(0..(self.m_payload_len + if self.m_crc_presence { 2 } else { 0 }));
        }

        Ok(())
    }
}
