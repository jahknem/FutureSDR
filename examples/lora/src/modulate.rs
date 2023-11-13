use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::f32::consts::PI;
use std::mem;
// use futuresdr::futures::FutureExt;
use futuresdr::futures::channel::mpsc;
use futuresdr::futures::executor::block_on;
use futuresdr::futures_lite::StreamExt;
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

use rustfft::{FftDirection, FftPlanner};

pub struct Modulate {
    m_sf: usize,                 // Transmission spreading factor
    m_samp_rate: usize,          // Transmission sampling rate
    m_bw: usize,                 // Transmission bandwidth (Works only for samp_rate=bw)
    m_number_of_bins: usize,     // number of bin per loar symbol
    m_samples_per_symbol: usize, // samples per symbols(Works only for 2^sf)
    m_sync_words: Vec<usize>,    // sync words (network id)

    m_ninput_items_required: usize, // number of samples required to call this block (forecast)

    m_os_factor: usize, // ovesampling factor based on sampling rate and bandwidth

    m_inter_frame_padding: usize, // length in samples of zero append to each frame

    m_frame_len: usize, // leng of the frame in number of items

    m_upchirp: Vec<Complex32>,   // reference upchirp
    m_downchirp: Vec<Complex32>, // reference downchirp

    m_preamb_len: usize,    // number of upchirps in the preamble
    samp_cnt: isize,        // counter of the number of lora samples sent
    preamb_samp_cnt: usize, // counter of the number of preamble symbols output
    padd_cnt: usize,        // counter of the number of null symbols output after each frame
    frame_cnt: u64,         // counter of the number of frame sent
    frame_end: bool,        // indicate that we send a full frame
}

impl Modulate {
    pub fn new(
        sf: usize,
        samp_rate: usize,
        bw: usize,
        sync_words: Vec<usize>,
        frame_zero_padd: usize,
        preamble_len: Option<usize>,
    ) -> Block {
        let preamble_len_tmp = preamble_len.unwrap_or(8);
        if preamble_len_tmp < 5 {
            panic!("Preamble length should be greater than 5!"); // only warning in original implementation
        }
        let sync_words_tmp: Vec<usize> = if sync_words.len() == 1 {
            let tmp = sync_words[0];
            vec![((tmp & 0xF0_usize) >> 4) << 3, (tmp & 0x0F_usize) << 3]
        } else {
            sync_words
        };
        let os_factor_tmp = samp_rate / bw;
        let number_of_bins_tmp = 1 << sf;
        let (ref_upchirp, ref_downchirp) = build_ref_chirps(sf, os_factor_tmp);

        Block::new(
            BlockMetaBuilder::new("Modulate").build(),
            StreamIoBuilder::new()
                .add_input::<usize>("in")
                .add_output::<Complex32>("out")
                .build(),
            MessageIoBuilder::new().build(),
            Modulate {
                m_sf: sf,
                m_samp_rate: samp_rate,
                m_bw: bw,
                m_number_of_bins: number_of_bins_tmp,
                m_sync_words: sync_words_tmp,
                m_os_factor: os_factor_tmp,
                m_samples_per_symbol: number_of_bins_tmp * os_factor_tmp,
                m_ninput_items_required: 1,
                m_inter_frame_padding: frame_zero_padd,
                m_upchirp: ref_upchirp,
                m_downchirp: ref_downchirp,
                frame_end: true,
                m_preamb_len: preamble_len_tmp,
                samp_cnt: -1,
                preamb_samp_cnt: 0,
                frame_cnt: 0,
                padd_cnt: frame_zero_padd,
                m_frame_len: 0, // implicit
            },
        )
        // set_output_multiple(m_samples_per_symbol);
    }

    fn set_sf(&mut self, sf: usize) {
        self.m_sf = sf;
        self.m_number_of_bins = 1 << self.m_sf;
        // self.m_os_factor = self.m_samp_rate / self.m_bw;  // noop, as both values didn't change
        self.m_samples_per_symbol = self.m_number_of_bins * self.m_os_factor;

        let (ref_upchirp, ref_downchirp) = build_ref_chirps(self.m_sf, self.m_os_factor);
        self.m_downchirp = ref_downchirp;
        self.m_upchirp = ref_upchirp;
    }
}

#[async_trait]
impl Kernel for Modulate {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input = sio.input(0).slice::<usize>();
        let out = sio.output(0).slice::<Complex32>();
        let mut nitems_to_process = input.len();
        let noutput_items: usize = out.len();
        let mut output_offset = 0;

        // set_output_multiple(m_samples_per_symbol);  // TODO

        let tags: Vec<(usize, usize)> = sio
            .input(0)
            .tags()
            .iter()
            .filter_map(|x| match x {
                ItemTag {
                    index,
                    tag: Tag::NamedAny(n, val),
                } => {
                    if n == "frame_len" {
                        match (**val).downcast_ref().unwrap() {
                            Pmt::Usize(frame_len) => Some((*index, *frame_len)),
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
                nitems_to_process = min(tags[0].0, noutput_items / self.m_samples_per_symbol);
            } else {
                if tags.len() >= 2 {
                    nitems_to_process = min(tags[1].0, noutput_items / self.m_samples_per_symbol);
                }
                if self.frame_end {
                    self.m_frame_len = tags[0].0;
                    sio.output(0).add_tag(
                        0,
                        Tag::NamedAny(
                            "frame_len".to_string(),
                            Box::new(Pmt::Usize(
                                ((self.m_frame_len as f32 + self.m_preamb_len as f32 + 4.25)
                                    * self.m_samples_per_symbol as f32)
                                    as usize
                                    + self.m_inter_frame_padding,
                            )),
                        ),
                    );
                    self.samp_cnt = -1;
                    self.preamb_samp_cnt = 0;
                    self.padd_cnt = 0;
                    self.frame_end = false;
                }
            }
        }

        if self.samp_cnt == -1
        // preamble
        {
            for i in 0..(noutput_items / self.m_samples_per_symbol) {
                if self.preamb_samp_cnt < (self.m_preamb_len + 5) * self.m_samples_per_symbol
                //should output preamble part
                {
                    if self.preamb_samp_cnt < (self.m_preamb_len * self.m_samples_per_symbol) {
                        //upchirps
                        out[output_offset..(output_offset + self.m_samples_per_symbol)]
                            .copy_from_slice(&self.m_upchirp)
                    } else if self.preamb_samp_cnt
                        == (self.m_preamb_len * self.m_samples_per_symbol)
                    {
                        //sync words
                        let sync_upchirp =
                            build_upchirp(self.m_sync_words[0], self.m_sf, self.m_os_factor);
                        out[output_offset..(output_offset + self.m_samples_per_symbol)]
                            .copy_from_slice(&sync_upchirp);
                    } else if self.preamb_samp_cnt
                        == (self.m_preamb_len + 1) * self.m_samples_per_symbol
                    {
                        let sync_upchirp =
                            build_upchirp(self.m_sync_words[1], self.m_sf, self.m_os_factor);
                        out[output_offset..(output_offset + self.m_samples_per_symbol)]
                            .copy_from_slice(&sync_upchirp);
                    } else if self.preamb_samp_cnt
                        < (self.m_preamb_len + 4) * self.m_samples_per_symbol
                    {
                        //2.25 downchirps
                        out[output_offset..(output_offset + self.m_samples_per_symbol)]
                            .copy_from_slice(&self.m_downchirp)
                    } else if self.preamb_samp_cnt
                        == (self.m_preamb_len + 4) * self.m_samples_per_symbol
                    {
                        out[output_offset..(output_offset + self.m_samples_per_symbol / 4)]
                            .copy_from_slice(&self.m_downchirp[0..(self.m_samples_per_symbol / 4)]);
                        //correct offset dur to quarter of downchirp
                        output_offset -= 3 * self.m_samples_per_symbol / 4;
                        self.samp_cnt = 0;
                    }
                    output_offset += self.m_samples_per_symbol;
                    self.preamb_samp_cnt += self.m_samples_per_symbol;
                }
            }
        }
        //output payload
        if self.samp_cnt < (self.m_frame_len * self.m_samples_per_symbol) as isize
            && self.samp_cnt > -1
        {
            nitems_to_process = min(
                nitems_to_process,
                (noutput_items - output_offset) / self.m_samples_per_symbol,
            );
            nitems_to_process = min(nitems_to_process, input.len());
            for i in 0..nitems_to_process {
                let data_upchirp = build_upchirp(input[i], self.m_sf, self.m_os_factor);
                out[output_offset..(output_offset + self.m_samples_per_symbol)]
                    .copy_from_slice(&data_upchirp);
                output_offset += self.m_samples_per_symbol;
                self.samp_cnt += self.m_samples_per_symbol as isize;
            }
        } else {
            nitems_to_process = 0;
        }
        //padd frame end with zeros
        if self.samp_cnt >= (self.m_frame_len * self.m_samples_per_symbol) as isize
            && self.samp_cnt
                < (self.m_frame_len * self.m_samples_per_symbol + self.m_inter_frame_padding)
                    as isize
        {
            self.m_ninput_items_required = 0;
            let padd_size = min(
                noutput_items - output_offset,
                self.m_frame_len * self.m_samples_per_symbol + self.m_inter_frame_padding
                    - self.samp_cnt as usize,
            );
            out[output_offset..(output_offset + padd_size)].fill(Complex32::new(0., 0.));
            self.samp_cnt += padd_size as isize;
            self.padd_cnt += padd_size;
            output_offset += padd_size;
        }
        if self.samp_cnt
            == (self.m_frame_len * self.m_samples_per_symbol + self.m_inter_frame_padding) as isize
        {
            self.samp_cnt += 1;
            self.frame_cnt += 1;
            self.m_ninput_items_required = 1;
            self.frame_end = true;
            // #ifdef GR_LORA_PRINT_INFO
            //                 std::cout << "Frame " << frame_cnt << " sent\n";
            // #endif
        }
        // if (nitems_to_process)
        //     std::cout << ninput_items[0] << " " << nitems_to_process << " " << output_offset << " " << noutput_items << std::endl;
        sio.input(0).consume(nitems_to_process);
        sio.output(0).produce(nitems_to_process);
        Ok(())
    }
}
