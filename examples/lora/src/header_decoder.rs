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

pub struct HeaderDecoder {
    m_impl_header: bool,  // Specify if we use an explicit or implicit header
    m_print_header: bool, // print or not header information in terminal
    m_payload_len: usize, // The payload length in bytes
    m_has_crc: bool,      // Specify the usage of a payload CRC
    m_cr: usize,          // Coding rate
    m_ldro_mode: bool,    // use low datarate optimisation
    // header_chk: u8,       // The header checksum received in the header
    pay_cnt: usize, // The number of payload nibbles received
    // nout: usize,          // The number of data nibbles to output
    is_header: bool, // Indicate that we need to decode the header
}

impl HeaderDecoder {
    pub fn new(
        impl_head: bool,
        cr: usize,
        pay_len: usize,
        has_crc: bool,
        ldro_mode: bool,
        print_header: bool,
    ) -> Block {
        Block::new(
            BlockMetaBuilder::new("HeaderDecoder").build(),
            StreamIoBuilder::new()
                .add_input::<u8>("in")
                .add_output::<u8>("out")
                .build(),
            MessageIoBuilder::new()
                .add_output("frame_info")
                .add_output("err")
                .build(),
            HeaderDecoder {
                m_impl_header: impl_head,
                m_print_header: print_header,
                m_cr: cr,
                m_payload_len: pay_len,
                m_has_crc: has_crc,
                m_ldro_mode: ldro_mode,
                pay_cnt: 0,
                is_header: false,
            },
        )
        // set_tag_propagation_policy(TPP_DONT); // TODO
    }

    async fn publish_frame_info(
        &self,
        sio: &mut StreamIo,
        mio: &mut MessageIo<Self>,
        cr: usize,
        pay_len: usize,
        crc: bool,
        ldro_mode: bool,
        err: bool,
    ) {
        let mut header_content: HashMap<String, Pmt> = HashMap::new();

        header_content.insert("cr".to_string(), Pmt::Usize(cr));
        header_content.insert("pay_len".to_string(), Pmt::Usize(pay_len));
        header_content.insert("crc".to_string(), Pmt::Bool(crc));
        header_content.insert("ldro_mode".to_string(), Pmt::Bool(ldro_mode));
        header_content.insert("err".to_string(), Pmt::Bool(err));
        mio.output_mut(0)
            .post(Pmt::MapStrPmt(header_content.clone()))
            .await;
        if !err {
            //don't propagate downstream that a frame was detected
            sio.output(0).add_tag(
                0,
                Tag::NamedAny(
                    "frame_info".to_string(),
                    Box::new(Pmt::MapStrPmt(header_content)),
                ),
            );
        }
    }
}

#[async_trait]
impl Kernel for HeaderDecoder {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        mio: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input = sio.input(0).slice::<u8>();
        let out = sio.output(0).slice::<u8>();
        let mut nitem_to_consume = input.len();
        let mut nitem_to_produce: usize = 0;

        let tags: Vec<(usize, &HashMap<String, Pmt>)> = sio
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
                            Pmt::MapStrPmt(map) => Some((*index, map)),
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
                nitem_to_consume = tags[0].0;
            } else {
                if tags.len() >= 2 {
                    nitem_to_consume = tags[1].0 - tags[0].0;
                }
                self.is_header = if let Pmt::Bool(tmp) = tags[0].1.get("is_header").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                // print("ac ishead: "<<is_header);
                if self.is_header {
                    self.pay_cnt = 0;
                }
            }
        }
        if self.is_header && nitem_to_consume < 5 && !self.m_impl_header
        //ensure to have a full PHY header to process
        {
            nitem_to_consume = 0;
        }
        nitem_to_consume = min(nitem_to_consume, min(input.len(), out.len()));
        if nitem_to_consume > 0 {
            if self.is_header {
                if self.m_impl_header {
                    //implicit header, all parameters should have been provided
                    self.publish_frame_info(
                        sio,
                        mio,
                        self.m_cr,
                        self.m_payload_len,
                        self.m_has_crc,
                        self.m_ldro_mode,
                        false,
                    )
                    .await;

                    for i in 0..nitem_to_consume {
                        //only output payload or CRC
                        if self.pay_cnt
                            < self.m_payload_len * 2 + if self.m_has_crc { 4 } else { 0 }
                        {
                            self.pay_cnt += 1;
                            out[nitem_to_produce] = input[i];
                            nitem_to_produce += 1;
                        }
                    }
                } else {
                    //explicit header to decode

                    self.m_payload_len = ((input[0] << 4) + input[1]) as usize;

                    self.m_has_crc = input[2] & 1 != 0;
                    self.m_cr = (input[2] >> 1) as usize;

                    let header_chk = ((input[3] & 1) << 4) + input[4];

                    //check header Checksum
                    let c4: u8 = (input[0] & 0b1000) >> 3
                        ^ (input[0] & 0b0100) >> 2
                        ^ (input[0] & 0b0010) >> 1
                        ^ (input[0] & 0b0001);
                    let c3: u8 = (input[0] & 0b1000) >> 3
                        ^ (input[1] & 0b1000) >> 3
                        ^ (input[1] & 0b0100) >> 2
                        ^ (input[1] & 0b0010) >> 1
                        ^ (input[2] & 0b0001);
                    let c2: u8 = (input[0] & 0b0100) >> 2
                        ^ (input[1] & 0b1000) >> 3
                        ^ (input[1] & 0b0001)
                        ^ (input[2] & 0b1000) >> 3
                        ^ (input[2] & 0b0010) >> 1;
                    let c1: u8 = (input[0] & 0b0010) >> 1
                        ^ (input[1] & 0b0100) >> 2
                        ^ (input[1] & 0b0001)
                        ^ (input[2] & 0b0100) >> 2
                        ^ (input[2] & 0b0010) >> 1
                        ^ (input[2] & 0b0001);
                    let c0: u8 = (input[0] & 0b0001)
                        ^ (input[1] & 0b0010) >> 1
                        ^ (input[2] & 0b1000) >> 3
                        ^ (input[2] & 0b0100) >> 2
                        ^ (input[2] & 0b0010) >> 1
                        ^ (input[2] & 0b0001);
                    if self.m_print_header {
                        info!("\n--------Header--------");
                        info!("Payload length: {}", self.m_payload_len);
                        info!("CRC presence:   {}", self.m_has_crc);
                        info!("Coding rate:    {}", self.m_cr);
                    }
                    let mut head_err =
                        header_chk - (c4 << 4) + (c3 << 3) + (c2 << 2) + (c1 << 1) + c0 != 0;
                    head_err = false; // TODO
                    if head_err || self.m_payload_len == 0 {
                        if self.m_print_header && head_err {
                            info!("input[0]: {:04b}", input[0]);
                            info!("input[1]: {:04b}", input[1]);
                            info!("input[2]: {:04b}", input[2]);
                            info!("input[3]: {:04b}", input[3]);
                            info!("input[4]: {:04b}", input[4]);
                            info!("c0: {}", c0);
                            info!("c1: {}", c1);
                            info!("c2: {}", c2);
                            info!("c3: {}", c3);
                            info!("c4: {}", c4);
                            info!("header_chk: {}", header_chk);
                            info!(
                                "(c4 << 4) + (c3 << 3) + (c2 << 2) + (c1 << 1) + c0: {}",
                                (c4 << 4) + (c3 << 3) + (c2 << 2) + (c1 << 1) + c0
                            );
                            warn!("Header checksum invalid!"); // TODO here
                        }
                        if self.m_print_header && self.m_payload_len == 0 {
                            warn!("Frame can not be empty!");
                            warn!("item to process= {}", nitem_to_consume);
                        }
                        mio.output_mut(1).post(Pmt::Bool(true));
                        head_err = true;
                        nitem_to_produce = 0;
                    } else {
                        if self.m_print_header {
                            info!("Header checksum valid!");
                        }
                        // #ifdef GRLORA_DEBUG
                        //                         std::cout << "should have " << (int)header_chk << std::endl;
                        //                         std::cout << "got: " << (int)(c4 << 4) + (c3 << 3) + (c2 << 2) + (c1 << 1) + c0 << std::endl;
                        // #endif
                        // nitem_to_produce = nitem_to_consume - HEADER_LEN;
                    }
                    self.publish_frame_info(
                        sio,
                        mio,
                        self.m_cr,
                        self.m_payload_len,
                        self.m_has_crc,
                        self.m_ldro_mode,
                        head_err,
                    )
                    .await;
                    // print("pub header info");
                    for i in HEADER_LEN..nitem_to_consume {
                        self.pay_cnt += 1;
                        out[nitem_to_produce] = input[i];
                        nitem_to_produce += 1;
                    }
                }
            } else {
                // info!("HeaderDecoder: asdf");
                // no header to decode
                for i in 0..nitem_to_consume {
                    // TODO can be simplified to range assignment
                    if self.pay_cnt < self.m_payload_len * 2 + if self.m_has_crc { 4 } else { 0 } {
                        //only output usefull value (payload and CRC if any)
                        self.pay_cnt += 1;
                        out[nitem_to_produce] = input[i];
                        nitem_to_produce += 1;
                    }
                }
                // nitem_to_consume = nitem_to_produce;
            }
            // if nitem_to_produce > 0 {
            //     info!("HeaderDecoder: producing {} samples", nitem_to_produce);
            // }
            // for i in 0..nitem_to_produce {
            //     info!("out[{}]: {:04b}", i, out[i]);
            // }
            sio.input(0).consume(nitem_to_consume);
            sio.output(0).produce(nitem_to_produce);
        }

        Ok(())
    }
}
