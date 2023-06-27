use async_net::{TcpListener, TcpStream};
use futures::AsyncReadExt;
use futures::AsyncWriteExt;
use futuresdr::log::{info, debug};
use std::cmp;

use futuresdr::anyhow::{bail, Context, Result};
use futuresdr::async_trait::async_trait;
use futuresdr::runtime::Block;
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::WorkIo;
use futuresdr::num_complex::Complex32;


use std::any::TypeId;

const TCP_EXCHANGER_PORT: u32 = 1592;

pub struct Complex32Serializer {
}
pub struct Complex32Deserializer {
}

impl Complex32Serializer {
    pub fn new() -> Block {
        Block::new(
            BlockMetaBuilder::new("Complex32Serializer").build(),
            StreamIoBuilder::new().add_input::<Complex32>("in").add_output::<u8>("out").build(),
            MessageIoBuilder::new().build(),
            Complex32Serializer {
            },
        )
    }
}
impl Complex32Deserializer {
    pub fn new() -> Block {
        Block::new(
            BlockMetaBuilder::new("Complex32Deserializer").build(),
            StreamIoBuilder::new().add_input::<u8>("in").add_output::<Complex32>("out").build(),
            MessageIoBuilder::new().build(),
            Complex32Deserializer {
            },
        )
    }
}

#[doc(hidden)]
#[async_trait]
impl Kernel for Complex32Serializer {
    async fn work(
        &mut self,
        io: &mut WorkIo,
        sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
    ) -> Result<()> {

        let mut out = sio.output(0).slice::<u8>();
        if out.is_empty() {
            return Ok(());
        }
        println!("out buffer len: {}", out.len());

        let input = sio.input(0).slice::<Complex32>();
        if input.len() > 0 {
            println!("received {} Complex32s", input.len());
            // convert Complex32 to bytes
            let n_input_to_produce = cmp::min(input.len(), out.len() / 8);
            if n_input_to_produce > 0 {
                let n_bytes_to_produce: usize = n_input_to_produce *8;
                for i in 0..n_input_to_produce {
                    out[i * 8..i * 8 + 4].copy_from_slice(&input[i].re.to_ne_bytes());
                    out[i * 8 + 4..i * 8 + 8].copy_from_slice(&input[i].im.to_ne_bytes());
                }
                sio.output(0).produce(n_bytes_to_produce);
                println!("converted {} Complex32s to {} bytes.", n_input_to_produce, n_bytes_to_produce);
            }

            if sio.input(0).finished() {
                io.finished = true;
            }

            sio.input(0).consume(n_input_to_produce);
        }


        Ok(())
    }

    async fn init(
        &mut self,
        _sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
    ) -> Result<()> {
        Ok(())
    }
}

#[doc(hidden)]
#[async_trait]
impl Kernel for Complex32Deserializer {
    async fn work(
        &mut self,
        io: &mut WorkIo,
        sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
    ) -> Result<()> {

        let mut out = sio.output(0).slice::<Complex32>();
        if out.is_empty() {
            return Ok(());
        }
        println!("out buffer len: {}", out.len());

        let input = sio.input(0).slice::<u8>();
        if input.len() > 0 {
            println!("received {} Bytes", input.len());
            // convert Complex32 to bytes
            let n_input_to_produce = cmp::min(input.len(), out.len() * 8);
            if n_input_to_produce > 0 {
                let n_output_to_produce: usize = n_input_to_produce / 8;
                for i in 0..n_output_to_produce {
                    out[i] = Complex32::new(
                        f32::from_ne_bytes(input[i * 8..i * 8 + 4].try_into().expect("does not happen")),
                        f32::from_ne_bytes(input[i * 8 + 4..i * 8 + 8].try_into().expect("does not happen")),
                    );
                }
                sio.output(0).produce(n_output_to_produce);
                println!("converted {} bytes to {} Complex32s.", n_input_to_produce, n_output_to_produce);
            }

            if sio.input(0).finished() {
                io.finished = true;
            }

            sio.input(0).consume(input.len());
        }

        Ok(())
    }

    async fn init(
        &mut self,
        _sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
    ) -> Result<()> {
        Ok(())
    }
}

