use async_net::{TcpListener, TcpStream};
use futures::AsyncReadExt;
use futures::AsyncWriteExt;
use futuresdr::log::{info, debug};

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
        debug!("out buffer len: {}", out.len());
        if out.is_empty() {
            return Ok(());
        }

        let input = sio.input(0).slice::<Complex32>();
        if input.len() > 0 {
            debug!("received {} Complex32s", input.len());
            // convert Complex32 to bytes
            let mut n_bytes: usize = 0;
            for i in 0..input.len() {
                out[i * 8..i * 8 + 4].copy_from_slice(&input[i].re.to_ne_bytes());
                out[i * 8 + 4..i * 8 + 8].copy_from_slice(&input[i].im.to_ne_bytes());
                n_bytes += 8;
            }
            sio.output(0).produce(n_bytes);


            if sio.input(0).finished() {
                io.finished = true;
            }

            debug!("converted {} Complex32s to {} bytes.", input.len(), n_bytes);
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

        let input = sio.input(0).slice::<u8>();

        if input.len() > 0 {
            debug!("received {} Bytes", input.len());
            // convert Complex32 to bytes
            let num_samples = input.len() / 8_usize;
            for i in 0..num_samples {
                out[i] = Complex32::new(
                    f32::from_ne_bytes(input[i * 8..i * 8 + 4].try_into().expect("does not happen")),
                    f32::from_ne_bytes(input[i * 8 + 4..i * 8 + 8].try_into().expect("does not happen")),
                );
            }
            sio.output(0).produce(out.len());


            if sio.input(0).finished() {
                io.finished = true;
            }

            debug!("converted {} bytes to {} Complex32s.", input.len(), out.len());
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

// thread 'FutureSDR: INFO - converted 0 Complex32s to 32768 bytes.
// smol-21' panicked at 'FutureSDR: INFO - Acting as IP tunnel from 192.168.42.10 to 192.168.42.11.
// assertion failed: `(left == right)`
//   left: `TypeId { t: 9589296137796154101 }`,
//  right: `TypeId { t: 5574462982184004571 }`', /usr/local/src/FutureSDR/src/runtime/stream_io.rs:229:9

