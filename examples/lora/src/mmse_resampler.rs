use crate::MmseFirInterpolator;
use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
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
use num_traits::Num;
use std::cmp::min;
use std::collections::HashMap;
use std::iter::Sum;
use std::ops::Mul;
use std::string::String;

pub struct MmseResampler<T>
where
    T: 'static,
{
    d_mu: f32,
    d_mu_inc: f32,
    d_resamp: MmseFirInterpolator<'static, T>,
}

impl<T> MmseResampler<T>
where
    T: Copy + Send + Sync + Num + Sum<T> + Mul<f32, Output = T> + 'static,
{
    pub fn new(phase_shift: f32, resamp_ratio: f32) -> Block {
        Block::new(
            BlockMetaBuilder::new("MmseResampler").build(),
            StreamIoBuilder::new()
                .add_input::<T>("in")
                .add_output::<T>("out")
                .build(),
            MessageIoBuilder::new().build(),
            MmseResampler::<T> {
                d_mu: phase_shift,
                d_mu_inc: resamp_ratio,
                d_resamp: MmseFirInterpolator::<T>::new(),
            },
        )
        // set_tag_propagation_policy(TPP_ONE_TO_ONE); // TODO
    }
}

#[async_trait]
impl<T: Copy + Send + Sync + Num + Sum<T> + Mul<f32, Output = T> + 'static> Kernel
    for MmseResampler<T>
{
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        // if let Ok(guard) = self.guard.try_lock() {
        let input = sio.input(0).slice::<T>();
        let ninput_items = input
            .len()
            .saturating_sub(MmseFirInterpolator::<T>::get_n_lookahead());
        let out = sio.output(0).slice::<T>();
        let noutput_items = out.len();
        let nitem_to_process = min(noutput_items, ninput_items);
        if nitem_to_process > 0 {
            let mut ii: usize = 0; // input index
            let mut oo: usize = 0; // output index

            while ii < ninput_items && oo < noutput_items {
                out[oo] = self.d_resamp.interpolate(
                    &input[ii..(ii + MmseFirInterpolator::<T>::get_n_lookahead())],
                    self.d_mu,
                );
                oo += 1;

                let s = self.d_mu + self.d_mu_inc;
                let f = s.floor();
                let incr = f as usize;
                self.d_mu = s - f;
                ii += incr;
            }
            sio.input(0).consume(ii);
            sio.output(0).produce(oo);
            // println!("MmseResampler: consumed {}, produced {} samples", ii, oo);
        }
        Ok(())
    }
}
