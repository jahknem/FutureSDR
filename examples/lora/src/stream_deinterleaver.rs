use futuresdr::anyhow::Result;
use futuresdr::macros::async_trait;
use futuresdr::runtime::Block;
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::WorkIo;
use std::cmp::min;
use std::marker::PhantomData;

pub struct StreamDeinterleaver<'a, T> {
    num_out: usize, // number of output streams
    // out_idx: usize, // index of the output stream receiving the next input sample
    phantom: PhantomData<&'a T>,
    // guard: Mutex<bool>,
}

impl<T> StreamDeinterleaver<'_, T>
where
    T: Copy + Sync + 'static,
{
    pub fn new(num_outputs: usize) -> Block {
        let mut sio = StreamIoBuilder::new().add_input::<T>("in");
        for i in 0..num_outputs {
            sio = sio.add_output::<T>(&format!("out{}", i));
        }
        Block::new(
            BlockMetaBuilder::new("StreamDeinterleaver").build(),
            sio.build(),
            MessageIoBuilder::new().build(),
            StreamDeinterleaver::<'static, T> {
                num_out: num_outputs,
                // out_idx: 0,
                phantom: PhantomData,
            },
        )
        // set_tag_propagation_policy(TPP_ONE_TO_ONE); // TODO
    }
}

#[async_trait]
impl<T: Copy + Sync + 'static> Kernel for StreamDeinterleaver<'_, T> {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        // if let Ok(guard) = self.guard.try_lock() {
        let input = sio.input(0).slice::<T>();
        let nitem_to_consume = input.len();
        let n_items_to_produce = sio
            .outputs_mut()
            .iter_mut()
            .map(|x| x.slice::<T>().len())
            .min()
            .unwrap();
        let nitem_to_process = min(n_items_to_produce, nitem_to_consume / self.num_out);
        if nitem_to_process > 0 {
            for j in 0..self.num_out {
                let out = sio.output(j).slice::<T>();
                // let items_to_output: Vec<T> = input[j..]
                //     .iter()
                //     .step_by(self.num_out)
                //     .take(nitem_to_process)
                //     .copied()
                //     .collect();
                // TODO replace with .zip(out.iter_mut())
                // out[0..nitem_to_process].copy_from_slice(&items_to_output);
                for (out_slot, &in_item) in out[0..nitem_to_process].iter_mut().zip(
                    input[j..]
                        .iter()
                        .step_by(self.num_out)
                        .take(nitem_to_process),
                ) {
                    *out_slot = in_item;
                }
                // out[0..nitem_to_produce]
                //     .copy_from_slice(&input[0..nitem_to_consume].iter().step_by(self.num_out).to);
                // self.out_idx += 1;
                // self.out_idx %= self.num_out;
                sio.output(j).produce(nitem_to_process);
            }
            sio.input(0).consume(nitem_to_process * self.num_out);
        }
        // }
        Ok(())
    }
}
