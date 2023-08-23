use futures::AsyncReadExt;
use futures::AsyncWriteExt;
use futuresdr::log::{debug, info};
use std::cmp;

use futuresdr::anyhow::{bail, Context, Result};
use futuresdr::async_trait::async_trait;
use futuresdr::num_complex::Complex32;
use futuresdr::runtime::Block;
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::WorkIo;
use rand::prelude::thread_rng;
use rand_distr::{Distribution, Normal};
use std::f32::consts::SQRT_2;

// use std::any::TypeId;
// use std::num::FpCategory::Normal;
// use std::ops::Add;

// pub struct AWGN<T> where T: Copy + Add, Standard: Distribution<T>{
pub struct AWGNComplex32 {
    power: f32,
    power_sqrt: f32,
    distribution: Normal<f32>,
}

// impl AWGN<T> where T: Copy + Add {
impl AWGNComplex32 {
    pub fn new(power: f32) -> Block {
        Block::new(
            // BlockMetaBuilder::new("AWGN").build(),
            BlockMetaBuilder::new("AWGNComplex32").build(),
            // StreamIoBuilder::new().add_input::<T>("in").add_output::<T>("out").build(),
            StreamIoBuilder::new()
                .add_input::<Complex32>("in")
                .add_output::<Complex32>("out")
                .build(),
            MessageIoBuilder::new().build(),
            // AWGN<T> {
            AWGNComplex32 {
                power: power,
                power_sqrt: power.sqrt(),
                distribution: Normal::new(0.0, SQRT_2 / 2.0).unwrap(),
            },
        )
    }

    // fn new_noise_sample() -> T {
    fn new_noise_sample(&self) -> Complex32 {
        // https://www.wavewalkerdsp.com/2022/06/01/how-to-create-additive-white-gaussian-noise-awgn/
        let mut rng = thread_rng();
        let real = self.distribution.sample(&mut rng);
        let imag = self.distribution.sample(&mut rng);
        let sample = Complex32::new(real, imag);
        let sample = self.power_sqrt * sample;
        return sample;
    }
}

#[doc(hidden)]
#[async_trait]
impl Kernel for AWGNComplex32 {
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
        debug!("out buffer len: {}", out.len());

        let input = sio.input(0).slice::<Complex32>();
        if input.len() > 0 {
            debug!("received {} Complex32s", input.len());
            // convert Complex32 to bytes
            let n_samples_to_produce = cmp::min(input.len(), out.len());
            if n_samples_to_produce > 0 {
                for i in 0..n_samples_to_produce {
                    out[i] = &input[i] + self.new_noise_sample();
                }
                sio.output(0).produce(n_samples_to_produce);
                debug!("added white noise to {} samples.", n_samples_to_produce);
            }

            if sio.input(0).finished() {
                io.finished = true;
            }

            sio.input(0).consume(n_samples_to_produce);
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
