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
use std::collections::HashMap;

/// A block that computes the phase difference between two complex signals.
pub struct PhaseDifference {
    fft_size: usize,
    m_cfo_int: usize,
    m_sto_frac: f64,
    k_hat: usize,
    total_consumed_samples: usize,
}

impl PhaseDifference {
    /// Create a new instance of the PhaseDifference block.
    pub fn new(fft_size: usize) -> Block {
        Block::new(
            BlockMetaBuilder::new("PhaseDifference").build(),
            StreamIoBuilder::new().build(),
            MessageIoBuilder::new()
                .add_input("phase_info1", PhaseDifference::phase_info_handler1)
                .add_input("phase_info2", PhaseDifference::phase_info_handler2)
                .add_output("out")
                .build(),
            PhaseDifference {
                 fft_size,
                 m_cfo_int: 0,
                 m_sto_frac: 0.0,
                 k_hat: 0,
                 total_consumed_samples: 0,
            },
        )
    }
    fn store_phase_info(&mut self, p: Pmt) {
        if let Pmt::MapStrPmt(frame_info) = p {
            self.m_cfo_int = if let Pmt::Usize(temp) = frame_info.get("cr").unwrap() {
                *temp
            } else {
                panic!("invalid cr")
            };
            self.m_sto_frac = if let Pmt::F64(temp) = frame_info.get("sto_frac").unwrap() {
                *temp
            } else {
                panic!("invalid sto_frac")
            };
            self.k_hat = if let Pmt::Usize(temp) = frame_info.get("k_hat").unwrap() {
                *temp
            } else {
                panic!("invalid k_hat")
            };
            self.total_consumed_samples = if let Pmt::Usize(temp) = frame_info.get("total_consumed_samples").unwrap() {
                *temp
            } else {
                panic!("Missing consumed samples");
            };
        }
    }


    fn output_metadata(&self, mio: &mut MessageIo<Self>) {
        let mut metadata = HashMap::new();
        metadata.insert("m_cfo_int".to_string(), Pmt::Usize(self.m_cfo_int));
        metadata.insert("m_sto_frac".to_string(), Pmt::F64(self.m_sto_frac));
        metadata.insert("k_hat".to_string(), Pmt::Usize(self.k_hat));
        metadata.insert("total_consumed_samples".to_string(), Pmt::Usize(self.total_consumed_samples));
        
        mio.post(0, Pmt::MapStrPmt(metadata));
    }

    #[message_handler]
    fn phase_info_handler2(
        &mut self,
        _io: &mut WorkIo,
        mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
        p: Pmt,
    ) -> Result<Pmt> {
        self.store_phase_info(p);
        self.output_metadata(mio);
        Ok(Pmt::Null)
    }

    #[message_handler]
    fn phase_info_handler1(
        &mut self,
        _io: &mut WorkIo,
        mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
        p: Pmt,
    ) -> Result<Pmt> {
        self.store_phase_info(p);
        self.output_metadata(mio);
        Ok(Pmt::Null)
    }
}

#[async_trait]
impl Kernel for PhaseDifference {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        _sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let n = std::cmp::min(input0.len(), input1.len());
        let n = std::cmp::min(n, out.len());

        for i in 0..n {
            let phase0 = input0[i].arg();
            let phase1 = input1[i].arg();
            out[i] = Complex::new(phase0 - phase1, 0.0);
        }

        sio.input(0).consume(n);
        sio.input(1).consume(n);
        sio.output(0).produce(n);

        Ok(())
    }
}