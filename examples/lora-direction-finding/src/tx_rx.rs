use std::time;

use futures::channel::mpsc;
use futures::executor::block_on;
use futuresdr::anyhow::Result;
use futuresdr::async_io::Timer;
use futuresdr::blocks::Delay;
use futuresdr::blocks::FirBuilder;
use futuresdr::blocks::NullSink;
use futuresdr::blocks::MessagePipe;
use futuresdr::blocks::MessageSourceBuilder;
use futuresdr::blocks::Split;
use futuresdr::macros::connect;
use futuresdr::num_complex::Complex32;
use futuresdr::runtime::buffer::circular::Circular;
use futuresdr::runtime::Flowgraph;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::Runtime;
use futures::StreamExt;
use lora::frame_sync;
use lora::whitening;
use futuresdr::gui::Gui;
use futuresdr::gui::GuiFrontend;
use lora_direction_finding::phase_difference;
use lora_direction_finding::phase_difference::PhaseDifference;
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
        Pmt::String("foooooooooooooooooooooooooo".to_string()),
        time::Duration::from_millis(500),
    )
    .n_messages(20000)
    .build();

    // GUI
    let spectrum = SpectrumPlotBuilder::new(args.bandwidth as f64)
        .center_frequency(args.frequency)
        .fft_size(2048)
        .build();

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

    const STO_FRAC_DENOM: isize = 1000; // How much to upsample
    const STO_FRAC_NOM: isize = 0; // Delay in sample parts (defined by denom)
    const STO_INT: isize = 0; // Delay in ganzen samples



    let up_sample_a = FirBuilder::new_resampling::<Complex32, Complex32>(STO_FRAC_DENOM.abs() as usize, 1); // 
    let sampling_time_offset_a = Delay::<Complex32>::new((STO_INT + 23) * STO_FRAC_DENOM + STO_FRAC_NOM - 1); // -1 to compensate resampling delay of 1 sample (I guess...)
    let down_sample_a = FirBuilder::new_resampling::<Complex32, Complex32>(1, STO_FRAC_DENOM.abs() as usize); // Bruchteil nehmen gemäß des os_factor in frame_sync


    let up_sample_b = FirBuilder::new_resampling::<Complex32, Complex32>(STO_FRAC_DENOM.abs() as usize, 1); // 
    let sampling_time_offset_b = Delay::<Complex32>::new((0 + 23) * STO_FRAC_DENOM + 0 - 1); // -1 to compensate resampling delay of 1 sample (I guess...)
    let down_sample_b = FirBuilder::new_resampling::<Complex32, Complex32>(1, STO_FRAC_DENOM.abs() as usize); // Bruchteil nehmen gemäß des os_factor in frame_sync 


    // RX chain A
    let frame_sync_a = FrameSync::new(
        args.frequency as u32,
        args.bandwidth as u32,
        args.spreading_factor,
        false,
        vec![0x12],
        1, // auf 4 setzen
        None,
    );
    let fft_demod_a = FftDemod::new(soft_decoding, true, args.spreading_factor);
    let gray_mapping_a = GrayMapping::new(soft_decoding);
    let deinterleaver_a = Deinterleaver::new(soft_decoding);
    let hamming_dec_a = HammingDec::new(soft_decoding);
    let header_decoder_a = HeaderDecoder::new(HeaderMode::Explicit, false);
    let decoder_a = Decoder::new();
    let null_sink_a = NullSink::<f32>::new();
    let (sender_a, mut receiver_a) = mpsc::channel::<Pmt>(10);
    let channel_sink_a = MessagePipe::new(sender_a);

    //RX Chain B
    let frame_sync_b = FrameSync::new(
        args.frequency as u32,
        args.bandwidth as u32,
        args.spreading_factor,
        false,
        vec![0x12],
        1, // auf 4 setzen
        None,
    );
    let fft_demod_b = FftDemod::new(soft_decoding, true, args.spreading_factor);
    let gray_mapping_b = GrayMapping::new(soft_decoding);
    let deinterleaver_b = Deinterleaver::new(soft_decoding);
    let hamming_dec_b = HammingDec::new(soft_decoding);
    let header_decoder_b = HeaderDecoder::new(HeaderMode::Explicit, false);
    let decoder_b = Decoder::new();
    let null_sink_b = NullSink::<f32>::new();
    let (sender_b, mut receiver_b) = mpsc::channel::<Pmt>(10);
    let channel_sink_b = MessagePipe::new(sender_b);

    // Shared RX Chain
    let phase_difference = PhaseDifference::new(2048);


    // TX Connect Macro
    connect!(
        fg, 
        msg_source | whitening.msg;
        whitening > header > add_crc > hamming_enc > interleaver > gray_demap > modulate
    );
    let split_function = |input: &Complex32| -> (Complex32, Complex32) {
        (*input, *input)  // Example split logic
    };
    let split_block = Split::new(split_function);
    let complex_null_sink_0 = NullSink::<Complex32>::new();
    let complex_null_sink_1 = NullSink::<Complex32>::new();

    connect!(
        fg, 
        modulate [Circular::with_size(2 * 4 * 8192 * 4 * 2)] split_block.in;
        split_block.out0 > 
        up_sample_a [Circular::with_size(2 * 4 * 8192 * 4 * 2)] sampling_time_offset_a [Circular::with_size(2 * 4 * 8192 * 4 * 2)] down_sample_a > 
        frame_sync_a;
        split_block.out1 > 
        up_sample_b [Circular::with_size(2 * 4 * 8192 * 4 * 2)] sampling_time_offset_b [Circular::with_size(2 * 4 * 8192 * 4 * 2)] down_sample_b > 
        frame_sync_b;
    );
    
    // connect!(
    //     fg, 
    //     modulate [Circular::with_size(2 * 4 * 8192 * 4 * 2)] frame_sync
    // );

    connect!(
        fg,
        frame_sync_a > fft_demod_a > gray_mapping_a > deinterleaver_a > hamming_dec_a > header_decoder_a;
        frame_sync_a.log_out > null_sink_a; 
        header_decoder_a.frame_info | frame_sync_a.frame_info; 
        header_decoder_a | decoder_a;
        decoder_a.data | channel_sink_a;
        frame_sync_b > fft_demod_b > gray_mapping_b > deinterleaver_b > hamming_dec_b > header_decoder_b;
        frame_sync_b.log_out > null_sink_b;
        header_decoder_b.frame_info | frame_sync_b.frame_info;
        header_decoder_b | decoder_b;
        decoder_b.data | channel_sink_b;
        frame_sync_a.phase_info | phase_difference.phase_info1;
        frame_sync_b.phase_info | phase_difference.phase_info2;
    );

    rt.spawn_background(async move {
        while let Some(x) = receiver_a.next().await {
            println!("Received: {:?}", x)
        }
    });

    // let (_fg, handle) = block_on(rt.start(fg));


    let _ = rt.run(fg);

    // Auskommentiert da dafür der gui branch von Felix notwendig ist.
    // connect!(fg,
    //     modulate > spectrum;
    // );
    // Gui::run(fg);

    Ok(())
}