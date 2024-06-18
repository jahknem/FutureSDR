use std::time;

use futures::channel::mpsc;
use futures::executor::block_on;
use futuresdr::anyhow::Result;
use futuresdr::async_io::Timer;
use futuresdr::blocks::NullSink;
use futuresdr::blocks::MessagePipe;
use futuresdr::blocks::MessageSourceBuilder;
use futuresdr::macros::connect;
use futuresdr::num_complex::Complex32;
use futuresdr::runtime::buffer::circular::Circular;
use futuresdr::runtime::Flowgraph;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::Runtime;
use futures::StreamExt;
use lora::frame_sync;
use lora::whitening;
// use futuresdr::gui::Gui;
// use futuresdr::gui::GuiFrontend;
use futuresdr::blocks::gui::SpectrumPlotBuilder;
use lora::{
    AddCrc, Decoder, Deinterleaver, FftDemod, FrameSync, GrayDemap, GrayMapping, HammingDec,
    HammingEnc, Header, HeaderDecoder, HeaderMode, Interleaver, Modulate, Whitening,
};
use std::time::Duration;
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    /// send periodic messages for testing
    #[clap(long, value_parser)]
    tx_interval: Option<f32>,
    /// lora spreading factor
    #[clap(long, default_value_t = 7)]
    spreading_factor: usize,
    /// lora bandwidth
    #[clap(long, default_value_t = 125000)]
    bandwidth: usize,
    /// lora sync word
    #[clap(long, default_value_t = 0x12)]
    sync_word: u8,
    /// lora frequency
    #[clap(long, default_value_t = 868.1e6)]
    frequency: f64, // Add the frequency field!
    /// lora amplitude
    #[clap(long, default_value_t = 1.0)]
    lora_amplitude: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let rt = Runtime::new();
    let mut fg = Flowgraph::new();

    // TX/RX Config
    let soft_decoding: bool = false;

    let msg_source = MessageSourceBuilder::new(
        Pmt::String("foo".to_string()),
        time::Duration::from_millis(100),
    )
    .n_messages(20000)
    .build();

    // GUI
    // let spectrum = SpectrumPlotBuilder::new(args.bandwidth as f64)
    //     .center_frequency(args.frequency)
    //     .fft_size(2048)
    //     .build();

    // TX Chain

    let has_crc = true;
    let cr = 0;

    let whitening = Whitening::new(false, false);
    let header = Header::new(false, has_crc, cr);
    let add_crc = AddCrc::new(has_crc);
    let hamming_enc = HammingEnc::new(cr, args.spreading_factor);
    let interleaver = Interleaver::new(cr as usize, args.spreading_factor, 0, args.bandwidth);
    let gray_demap = GrayDemap::new(args.spreading_factor);
    let intermediate_sample_rate = args.bandwidth as usize * 1;
    let modulate = Modulate::new(
        args.spreading_factor,
        intermediate_sample_rate,
        args.bandwidth as usize,
        vec![0x12],
        20 * (1 << args.spreading_factor) * intermediate_sample_rate / args.bandwidth,
        None,
    );

    // TX Connect Macro
    connect!(
        fg, 
        msg_source | whitening.msg;
        whitening > header > add_crc > hamming_enc > interleaver > gray_demap > modulate
    );


    // RX chain
    let frame_sync = FrameSync::new(
        args.frequency as u32,
        args.bandwidth as u32,
        args.spreading_factor,
        false,
        vec![0x12],
        1,
        None,
    );
    let fft_demod = FftDemod::new(soft_decoding, true, args.spreading_factor);
    let gray_mapping = GrayMapping::new(soft_decoding);
    let deinterleaver = Deinterleaver::new(soft_decoding);
    let hamming_dec = HammingDec::new(soft_decoding);
    let header_decoder = HeaderDecoder::new(HeaderMode::Explicit, false);
    let decoder = Decoder::new();
    let null_sink = NullSink::<f32>::new();
    let (sender, mut receiver) = mpsc::channel::<Pmt>(10);
    let channel_sink = MessagePipe::new(sender);

    connect!(fg, modulate [Circular::with_size(2 * 4 * 8192 * 4 * 2)] frame_sync);

    connect!(fg,
        frame_sync > fft_demod > gray_mapping > deinterleaver > hamming_dec > header_decoder;
        frame_sync.log_out > null_sink; 
        header_decoder.frame_info | frame_sync.frame_info; 
        header_decoder | decoder;
        decoder.data | channel_sink;
    );

    rt.spawn_background(async move {
        while let Some(x) = receiver.next().await {
            println!("Received: {:?}", x)
        }
    });

    // let (_fg, handle) = block_on(rt.start(fg));

    // Auskommentiert da dafür der gui branch von Felix notwendig ist.
    let _ = rt.run(fg);

    // connect!(fg,
    //     modulate > spectrum;
    // );
    // Gui::run(fg);

    Ok(())
}