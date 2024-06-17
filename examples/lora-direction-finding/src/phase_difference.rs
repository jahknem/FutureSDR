use futuresdr::anyhow::Result;
use futuresdr::macros::async_trait;
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::WorkIo;

use rustfft::num_complex::Complex;

pub struct PhaseDifference {
    fft_size: usize,
}

impl PhaseDifference {
    pub fn new(fft_size: usize) -> Block {
        Block::new(
            BlockMetaBuilder::new("PhaseDifference").build(),
            StreamIoBuilder::new()
                .add_input::<Complex<f32>>("in0")
                .add_input::<Complex<f32>>("in1")
                .add_output::<f32>("out")
                .build(),
            MessageIoBuilder::new().build(),
            PhaseDifference { fft_size },
        )
    }
}

#[async_trait]
impl Kernel for PhaseDifference {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input0 = sio.input(0).slice::<Complex<f32>>();
        let input1 = sio.input(1).slice::<Complex<f32>>();
        let out = sio.output(0).slice::<f32>();

        let n = std::cmp::min(input0.len(), input1.len());
        let n = std::cmp::min(n, out.len());

        for i in 0..n {
            let phase0 = input0[i].arg();
            let phase1 = input1[i].arg();
            out[i] = phase0 - phase1;
        }

        sio.input(0).consume(n);
        sio.input(1).consume(n);
        sio.output(0).produce(n);

        Ok(())
    }
}