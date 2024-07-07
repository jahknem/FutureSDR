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
use futuresdr::macros::message_handler;
use futuresdr::runtime::Block;
use futuresdr::runtime::Pmt;
use futuresdr::log::{warn, info, debug};

use rustfft::num_complex::Complex;

pub struct PhaseDifference {
    fft_size: usize,
}

impl PhaseDifference {
    pub fn new(fft_size: usize) -> Block {
        Block::new(
            BlockMetaBuilder::new("PhaseDifference").build(),
            StreamIoBuilder::new().build(),
            MessageIoBuilder::new()
                .add_input("in1")
                .add_input("in2")
                .add_output("out")
                .build(),
            PhaseDifference {
                 fft_size 
            },
        )
    }
    fn store_phase_info(&mut self, p: Pmt) {
        if let Pmt::MapStrPmt(mut frame_info) = p {
            let m_cfo_int: usize = if let Pmt::Usize(temp) = frame_info.get("cr").unwrap() {
                *temp
            } else {
                panic!("invalid cr")
            };
            let m_sto_frac: f64 = if let Pmt::F64(temp) = frame_info.get("sto_frac").unwrap() {
                *temp
            } else {
                panic!("invalid sto_frac")
            };
            let k_hat: usize = if let Pmt::Usize(temp) = frame_info.get("k_hat").unwrap() {
                *temp
            } else {
                panic!("invalid k_hat")
            };
            let total_consumed_samples: usize = if let Pmt::Usize(temp) = frame_info.get("total_consumed_samples").unwrap() {
                *temp
            } else {
                panic!("Missing consumed samples");
            };
    #[message_handler]
    fn phase_info_handler2(
        &mut self,
        _io: &mut WorkIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
        p: Pmt,
    ) -> Result<Pmt> {
        store_phase_info(p);

        Ok(Pmt::Null)
    }
    #[message_handler]
    fn phase_info_handler1(
        &mut self,
        _io: &mut WorkIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
        p: Pmt,
    ) -> Result<Pmt> {

    if let Pmt::MapStrPmt(mut frame_info) = p {
        let m_cfo_int: usize = if let Pmt::Usize(temp) = frame_info.get("cr").unwrap() {
            *temp
        } else {
            panic!("invalid cr")
        };
        let m_sto_frac: f64 = if let Pmt::F64(temp) = frame_info.get("sto_frac").unwrap() {
            *temp
        } else {
            panic!("invalid sto_frac")
        };
        let k_hat: usize = if let Pmt::Usize(temp) = frame_info.get("k_hat").unwrap() {
            *temp
        } else {
            panic!("invalid k_hat")
        };
        let total_consumed_samples: usize = if let Pmt::Usize(temp) = frame_info.get("total_consumed_samples").unwrap() {
            *temp
        } else {
            panic!("Missing consumed samples");
        };


        let m_invalid_header = if let Pmt::Bool(temp) = frame_info.get("err").unwrap() {
            *temp
        } else {
            panic!("invalid err flag")
        };

        Ok(Pmt::Null)
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
        let input0 = sio.input(0).slice::<Complex32>();
        let input1 = sio.input(1).slice::<Complex32>();
        let out = sio.output(0).slice::<Complex32>();

        

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