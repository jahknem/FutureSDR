use futuredsp::fir::NonResamplingFirKernel;
use futuredsp::UnaryKernel;
use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use futuresdr::log::info;
use futuresdr::num_complex::Complex32;
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
use rustfft::{Fft, FftDirection, FftPlanner};
use std::cmp::min;
use std::sync::Arc;

use crate::utilities::*;

pub struct PfbChannelizer {
    d_updated: bool,
    // d_oversample_rate: f32,
    d_idxlut: Vec<usize>,
    // d_rate_ratio: usize,
    // d_output_multiple: i32,
    d_channel_map: Vec<usize>,
    // gr::thread::mutex d_mutex: ,
    d_fir_filters: Vec<NonResamplingFirKernel<Complex32, Complex32, Vec<f32>, f32>>,
    d_taps_per_filter: usize,
    d_filts: usize,
    d_fft: Arc<dyn Fft<f32>>,
}

impl PfbChannelizer {
    pub fn new(nfilts: usize, taps: &[f32], oversample_rate: f32) -> Block {
        // polyphase_filterbank(nfilts, taps),
        let srate: f64 = nfilts as f64 / oversample_rate as f64;
        let rsrate: f64 = srate.round();
        if (srate - rsrate).abs() > 0.00001 {
            panic!("pfb_channelizer: oversample rate must be N/i for i in [1, N]");
        }

        // set_relative_rate(oversample_rate);  // TODO

        // Default channel map. The channel map specifies which input
        // goes to which output channel; so out[0] comes from
        // channel_map[0].
        let channel_map: Vec<usize> = (0..nfilts).collect();

        // We use a look up table to set the index of the FFT input
        // buffer, which equivalently performs the FFT shift operation
        // on every other turn when the rate_ratio>1.  Also, this
        // performs the index 'flip' where the first input goes into the
        // last filter. In the pfb_decimator_ccf, we directly index the
        // input_items buffers starting with this last; here we start
        // with the first and put it into the fft object properly for
        // the same effect.
        let rate_ratio = (nfilts as f32 / oversample_rate) as usize;
        let idxlut: Vec<usize> = (0..nfilts)
            .map(|x| nfilts - ((x + rate_ratio) % nfilts) - 1)
            .collect();

        // Calculate the number of filtering rounds to do to evenly
        // align the input vectors with the output channels
        let mut output_multiple = 1;
        while (output_multiple * rate_ratio) % nfilts != 0 {
            output_multiple += 1;
        }
        // set_output_multiple(d_output_multiple);  // TODO

        let mut fir_filters: Vec<NonResamplingFirKernel<Complex32, Complex32, Vec<f32>, f32>> =
            vec![];
        let n_taps_per_filter = (taps.len() as f32 / nfilts as f32).ceil() as usize;
        for i in 0..nfilts {
            let mut taps_tmp: Vec<f32> = taps[i..].iter().step_by(nfilts).copied().collect();
            if taps_tmp.len() < n_taps_per_filter {
                taps_tmp.push(0.);
            }
            fir_filters.push(NonResamplingFirKernel::<Complex32, Complex32, _, _>::new(
                taps_tmp,
            ));
        }

        let mut channelizer = PfbChannelizer {
            d_updated: false,
            // d_oversample_rate: oversample_rate,
            d_channel_map: channel_map,
            d_idxlut: idxlut,
            // d_rate_ratio: rate_ratio,
            // d_output_multiple: output_multiple,
            d_fir_filters: fir_filters,
            d_taps_per_filter: n_taps_per_filter,
            d_filts: nfilts,
            d_fft: FftPlanner::new().plan_fft(nfilts, FftDirection::Inverse),
        };

        // Use set_taps to also set the history requirement
        channelizer.set_taps(taps);

        // because we need a stream_to_streams block for the input,
        // only send tags from in[i] -> out[i].
        // set_tag_propagation_policy(TPP_ONE_TO_ONE);  // TODO

        let mut sio = StreamIoBuilder::new();
        for i in 0..nfilts {
            sio = sio
                .add_input::<Complex32>(format!("in{i}").as_str())
                .add_output::<Complex32>(format!("out{i}").as_str());
        }

        Block::new(
            BlockMetaBuilder::new("PfbChannelizer").build(),
            sio.build(),
            MessageIoBuilder::new().build(),
            channelizer,
        )
    }

    fn set_taps(&mut self, taps: &[f32]) {
        // gr::thread::scoped_lock guard(d_mutex);

        // polyphase_filterbank::set_taps(taps);
        // set_history(d_taps_per_filter + 1);
        self.d_updated = true;
    }

    // fn print_taps() { polyphase_filterbank::print_taps(); }

    // fn taps() -> &[&[f32]]
    // {
    //     return polyphase_filterbank::taps();
    // }

    fn set_channel_map(&mut self, map: Vec<usize>) {
        // gr::thread::scoped_lock guard(d_mutex);

        if !map.is_empty() {
            let max = map.iter().max().unwrap();
            if *max >= self.d_filts {
                panic!("pfb_channelizer_ccf_impl::set_channel_map: map range out of bounds.");
            }
            self.d_channel_map = map;
        }
    }

    // std::vector<int> pfb_channelizer_ccf_impl::channel_map() const { return d_channel_map; }
}

#[async_trait]
impl Kernel for PfbChannelizer {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let n_items_available = sio
            .inputs_mut()
            .iter_mut()
            .map(|x| x.slice::<Complex32>().len())
            .min()
            .unwrap();
        let n_items_to_consume = n_items_available.saturating_sub(self.d_taps_per_filter);
        let n_items_to_produce = sio
            .outputs_mut()
            .iter_mut()
            .map(|x| x.slice::<Complex32>().len())
            .min()
            .unwrap();
        let n_items_to_process = min(n_items_to_produce, n_items_to_consume);

        if n_items_to_process > 0 {
            let mut fft_bufs: Vec<Vec<Complex32>> =
                vec![vec![Complex32::new(0., 0.); self.d_filts]; n_items_to_process];
            for i in 0..self.d_filts {
                let input = &sio.input(self.d_filts - 1 - i).slice::<Complex32>()
                    [0..(n_items_to_process + self.d_taps_per_filter)];
                let mut fir_output = vec![Complex32::new(0., 0.); n_items_to_process];
                let _ = self.d_fir_filters[i].work(input, &mut fir_output);
                for j in 0..n_items_to_process {
                    fft_bufs[j][i] = fir_output[j];
                }
            }

            let mut outs: Vec<&mut [Complex32]> = sio
                .outputs_mut()
                .iter_mut()
                .map(|x| x.slice::<Complex32>())
                .collect();
            for j in 0..n_items_to_process {
                self.d_fft.process(&mut fft_bufs[j]);
                for i in 0..self.d_filts {
                    outs[i][j] = fft_bufs[j][i]
                }
            }

            // println!("producing {} samples per channel", n_items_to_process);
            for i in 0..self.d_filts {
                sio.input(i).consume(n_items_to_process);
                sio.output(i).produce(n_items_to_process);
                // println!("PfbChannelizer: consumed {n_items_to_process}, produced {n_items_to_process} samples on channel {i}");
            }
        }
        // else if n_items_to_produce == 0 {
        //     println!("Flaggggaag!");
        // }

        // TODO propagate tags

        // let out = sio.output(0).slice::<Complex32>();
        // let mut nitems_to_process = input.len();
        // let noutput_items: usize = out.len();
        // let mut output_offset = 0;
        //
        // let tags: Vec<(usize, usize)> = sio
        //     .input(0)
        //     .tags()
        //     .iter()
        //     .filter_map(|x| match x {
        //         ItemTag {
        //             index,
        //             tag: Tag::NamedAny(n, val),
        //         } => {
        //             if n == "frame_len" {
        //                 match (**val).downcast_ref().unwrap() {
        //                     Pmt::Usize(frame_len) => Some((*index, *frame_len)),
        //                     _ => None,
        //                 }
        //             } else {
        //                 None
        //             }
        //         }
        //         _ => None,
        //     })
        //     .collect();
        // if !tags.is_empty() {
        //     if tags[0].0 != 0 {
        //         nitems_to_process = min(tags[0].0, noutput_items / self.m_samples_per_symbol);
        //         // info!(
        //         //     "Modulate: Flag 2 - nitems_to_process: {}",
        //         //     nitems_to_process
        //         // );
        //     } else {
        //         if tags.len() >= 2 {
        //             nitems_to_process = min(tags[1].0, noutput_items / self.m_samples_per_symbol);
        //             // info!(
        //             //     "Modulate: Flag 3 - nitems_to_process: {}",
        //             //     nitems_to_process
        //             // );
        //         }
        //         if self.frame_end {
        //             self.m_frame_len = tags[0].1;
        //             // sio.output(0).add_tag(
        //             //     0,
        //             //     Tag::NamedAny(
        //             //         "frame_len".to_string(),
        //             //         Box::new(Pmt::Usize(
        //             //             ((self.m_frame_len as f32 + self.m_preamb_len as f32 + 4.25)
        //             //                 * self.m_samples_per_symbol as f32)
        //             //                 as usize
        //             //                 + self.m_inter_frame_padding,
        //             //         )),
        //             //     ),
        //             // );
        //             sio.output(0).add_tag(
        //                 0,
        //                 Tag::NamedUsize(
        //                     "burst_start".to_string(),
        //                     ((self.m_frame_len as f32 + (self.m_preamb_len as f32 + 4.25))
        //                         * self.m_samples_per_symbol as f32)
        //                         as usize
        //                         + self.m_inter_frame_padding,
        //                 ),
        //             );
        //             self.samp_cnt = -1;
        //             self.preamb_samp_cnt = 0;
        //             self.padd_cnt = 0;
        //             self.frame_end = false;
        //         }
        //     }
        // }

        // // gr::thread::scoped_lock guard(d_mutex);
        //
        // gr_complex* in = (gr_complex*)input_items[0];
        // gr_complex* out = (gr_complex*)output_items[0];
        //
        // if (d_updated) {
        //     d_updated = false;
        //     return 0; // history requirements may have changed.
        // }
        //
        // size_t noutputs = output_items.size();
        //
        // // The following algorithm looks more complex in order to handle
        // // the cases where we want more that 1 sps for each
        // // channel. Otherwise, this would boil down into a single loop
        // // that operates from input_items[0] to [d_nfilts].
        //
        // // When dealing with osps>1, we start not at the last filter,
        // // but nfilts/osps and then wrap around to the next symbol into
        // // the other set of filters.
        // // For details of this operation, see:
        // // fred harris, Multirate Signal Processing For Communication
        // // Systems. Upper Saddle River, NJ: Prentice Hall, 2004.
        //
        // int n = 1, i = -1, j = 0, oo = 0, last;
        // int toconsume = (int)rintf(noutput_items / d_oversample_rate);
        // while (n <= toconsume) {
        //     j = 0;
        //     i = (i + d_rate_ratio) % d_nfilts;
        //     last = i;
        //     while (i >= 0) {
        //         in = (gr_complex*)input_items[j];
        //         d_fft.get_inbuf()[d_idxlut[j]] = d_fir_filters[i].filter(&in[n]);
        //         j++;
        //         i--;
        //     }
        //
        //     i = d_nfilts - 1;
        //     while (i > last) {
        //         in = (gr_complex*)input_items[j];
        //         d_fft.get_inbuf()[d_idxlut[j]] = d_fir_filters[i].filter(&in[n - 1]);
        //         j++;
        //         i--;
        //     }
        //
        //     n += (i + d_rate_ratio) >= (int)d_nfilts;
        //
        //     // despin through FFT
        //     d_fft.execute();
        //
        //     // Send to output channels
        //     for (unsigned int nn = 0; nn < noutputs; nn++) {
        //         out = (gr_complex*)output_items[nn];
        //         out[oo] = d_fft.get_outbuf()[d_channel_map[nn]];
        //     }
        //     oo++;
        // }
        //
        // consume_each(toconsume);
        // return noutput_items;

        Ok(())
    }
}
