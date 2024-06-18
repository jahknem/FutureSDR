use std::sync::Arc;

use futures::channel::mpsc;
use crate::mpsc::Receiver;
use seify::{Device, DeviceTrait, RxStreamer, TxStreamer};

use futuresdr::anyhow::Result;
use futuresdr::blocks::seify::SourceBuilder;
use futuresdr::blocks::NullSink;
use futuresdr::blocks::MessagePipe;
use futuresdr::macros::connect;
use futuresdr::runtime::buffer::circular::Circular;
use futuresdr::runtime::{Flowgraph, Pmt};

use lora::Decoder;
use lora::Deinterleaver;
use lora::FftDemod;
use lora::FrameSync;
use lora::GrayMapping;
use lora::HammingDec;
use lora::HeaderDecoder;
use lora::HeaderMode;

const SOFT_DECODING: bool = false;
const SPREADING_FACTOR: usize = 7;
const BANDWIDTH: f64 = 125_000.0;
const SYNC_WORD: u8 = 0x12;
const PAYLOAD_LENGTH: usize = 26;
const HEADER: HeaderMode = HeaderMode::Implicit {
    code_rate: 4,
    has_crc: false,
    payload_len: PAYLOAD_LENGTH,
};

fn find_usable_device() -> Result<Option<Device<Arc<dyn DeviceTrait<RxStreamer = Box<(dyn RxStreamer + 'static)>, TxStreamer = Box<(dyn TxStreamer + 'static)>> + Sync>>>> {
    for args in seify::enumerate()? {
        let device = seify::Device::from_args(args)?;
        let num_rx = device.num_channels(seify::Direction::Rx)?;
        println!("Device: {:?}", device.antennas(seify::Direction::Rx, 0));
        if num_rx >= 2 {
            return Ok(Some(device));
        }
    }

    return Ok(None)
}

pub fn add_lora_decoder(mut fg: &mut Flowgraph, sample_rate: f64, frequency: f64) -> Result<(usize, usize, Receiver<Pmt>)> {
    //let downsample =
    //    FirBuilder::new_resampling::<Complex32, Complex32>(1, (sample_rate / BANDWIDTH) as usize);
    let frame_sync = FrameSync::new(
        frequency as u32,
        BANDWIDTH as u32,
        SPREADING_FACTOR,
        false,
        vec![SYNC_WORD.into()],
        1,
        None,
    );
    let null_sink = NullSink::<f32>::new();
    let fft_demod = FftDemod::new(SOFT_DECODING, true, SPREADING_FACTOR);
    let gray_mapping = GrayMapping::new(SOFT_DECODING);
    let deinterleaver = Deinterleaver::new(SOFT_DECODING);
    let hamming_dec = HammingDec::new(SOFT_DECODING);
    let header_decoder = HeaderDecoder::new(HeaderMode::Explicit, false);
    let decoder = Decoder::new();

    let (sender, receiver) = mpsc::channel::<Pmt>(10);
    let channel_sink = MessagePipe::new(sender);

    connect!(fg,
             //downsample [Circular::with_size(2 * 4 * 8192 * 4)] frame_sync > fft_demod > gray_mapping > deinterleaver 
             frame_sync > fft_demod > gray_mapping > deinterleaver > hamming_dec > header_decoder;

            //  frame_sync > fft_demod > gray_mapping;
            //  gray_mapping.out_0 > deinterleaver > hamming_dec > header_decoder;
            //  gray_mapping.out_1 > deinterleaver > hamming_dec > header_decoder;
             frame_sync.log_out > null_sink;
            //  frame_sync.phase_out | phase_difference.in_0;
            //  frame_sync_2.phase_out | phase_difference.in_1;
             header_decoder.frame_info | frame_sync.frame_info;
             header_decoder.frame_info | channel_sink;
             header_decoder | decoder);

    //Ok((downsample, decoder))
    Ok((frame_sync, decoder, receiver))
}

pub fn build_flowgraph(
    sample_rate: f64,
    frequency: f64,
    gain: f64
) -> Result<(Flowgraph, Receiver<Pmt>, Receiver<Pmt>, Receiver<Pmt>, Receiver<Pmt>)> {
    let mut fg = Flowgraph::new();

    let device = find_usable_device()?.unwrap();

    let src = SourceBuilder::new()
        .device(device.clone())
        .channels(vec![0, 1])
    //    .sample_rate(sample_rate)
        .sample_rate(BANDWIDTH as f64)
        .antenna("TX/RX")
        .frequency(frequency)
        .gain(gain)
        .build()?;

    let (lora_dec_in1, lora_dec_out1, lora_frame_info_receiver1) = add_lora_decoder(&mut fg, sample_rate, frequency)?;
    let (lora_dec_in2, lora_dec_out2, lora_frame_info_receiver2) = add_lora_decoder(&mut fg, sample_rate, frequency)?;

    let (sender, receiver1) = mpsc::channel::<Pmt>(10);
    let channel_sink1 = MessagePipe::new(sender);

    let (sender, receiver2) = mpsc::channel::<Pmt>(10);
    let channel_sink2 = MessagePipe::new(sender);

    connect!(fg, src.out1 [Circular::with_size(2 * 4 * 8192 * 4)] lora_dec_in1; lora_dec_out1.data | channel_sink1);
    connect!(fg, src.out2 [Circular::with_size(2 * 4 * 8192 * 4)] lora_dec_in2; lora_dec_out2.data | channel_sink2);

    Ok((fg, receiver1, receiver2, lora_frame_info_receiver1, lora_frame_info_receiver2))
}
