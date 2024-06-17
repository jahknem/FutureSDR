use futuresdr::anyhow::Result;
use futuresdr::async_io::Timer;
use futuresdr::blocks::NullSink;
use futuresdr::macros::connect;
use futuresdr::num_complex::Complex32;
use futuresdr::runtime::buffer::circular::Circular;
use futuresdr::runtime::Flowgraph;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::Runtime;
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
    /// lora sample rate
    #[clap(long, default_value_t = 125000)]
    sample_rate: usize,
    /// lora sync word
    #[clap(long, default_value_t = 0x12)]
    sync_word: u8,
    /// lora frequency
    #[clap(long, default_value_t = 868.1e6)]
    frequency: f64, // Add the frequency field!
}

fn main() -> Result<()> {
    let args = Args::parse();

    let rt = Runtime::new();
    let mut fg = Flowgraph::new();
    let soft_decoding: bool = false;

    let null_sink = fg.add_block(NullSink::<Complex32>::new());
    let null_sink_rx = fg.add_block(NullSink::<f32>::new());

    // TX Chain
    let impl_head = false;
    let has_crc = true;
    let cr = 3;

    let whitening = Whitening::new(false, false);
    let fg_tx_port = whitening
        .message_input_name_to_id("msg")
        .expect("No message_in port found!");
    let whitening = fg.add_block(whitening);
    let header = fg.add_block(Header::new(impl_head, has_crc, cr));
    fg.connect_stream(whitening, "out", header, "in")?;
    let add_crc = fg.add_block(AddCrc::new(has_crc));
    fg.connect_stream(header, "out", add_crc, "in")?;
    let hamming_enc = fg.add_block(HammingEnc::new(cr, args.spreading_factor));
    fg.connect_stream(add_crc, "out", hamming_enc, "in")?;
    let interleaver = fg.add_block(Interleaver::new(
        cr as usize,
        args.spreading_factor,
        0,
        args.bandwidth,
    ));
    fg.connect_stream(hamming_enc, "out", interleaver, "in")?;
    let gray_demap = fg.add_block(GrayDemap::new(args.spreading_factor));
    fg.connect_stream(interleaver, "out", gray_demap, "in")?;
    let modulate = fg.add_block(Modulate::new(
        args.spreading_factor,
        args.sample_rate,
        args.bandwidth,
        vec![8, 16],
        20 * (1 << args.spreading_factor) * args.sample_rate / args.bandwidth,
        Some(8),
    ));
    fg.connect_stream(gray_demap, "out", modulate, "in")?;
    fg.connect_stream_with_type(
        modulate,
        "out",
        null_sink,
        "in",
        Circular::with_size(2 * 4 * 8192 * 4 * 8),
    )?;

    // RX chain
    let frame_sync = FrameSync::new(
        args.frequency as u32,
        args.bandwidth as u32,
        args.spreading_factor,
        false,
        vec![args.sync_word.into()],
        1,
        None,
    );
    let fft_demod = FftDemod::new(soft_decoding, true, args.spreading_factor);
    let gray_mapping = GrayMapping::new(soft_decoding);
    let deinterleaver = Deinterleaver::new(soft_decoding);
    let hamming_dec = HammingDec::new(soft_decoding);
    let header_decoder = HeaderDecoder::new(HeaderMode::Explicit, false);
    let decoder = Decoder::new();

    connect!(fg,
        whitening > header > add_crc > hamming_enc > interleaver > gray_demap > modulate 
            [Circular::with_size(2 * 4 * 8192 * 4 * 8)] null_sink;
        modulate [Circular::with_size(2 * 4 * 8192 * 4 * 8)] frame_sync > fft_demod 
            > gray_mapping > deinterleaver > hamming_dec > header_decoder;
        frame_sync.log_out > null_sink_rx;
        header_decoder.frame_info | frame_sync.frame_info; 
        header_decoder > decoder
    );

    // if tx_interval is set, send messages periodically
    if let Some(tx_interval) = args.tx_interval {
        let (_fg, mut handle) = rt.start_sync(fg);
        rt.block_on(async move {
            let mut counter: usize = 0;
            loop {
                Timer::after(Duration::from_secs_f32(tx_interval)).await;
                let dummy_packet = format!("hello world! {:02}", counter).to_string();
                // let dummy_packet = "hello world!1".to_string();
                handle
                    .call(whitening, fg_tx_port, Pmt::String(dummy_packet))
                    .await
                    .unwrap();
                println!("sending sample packet.");
                counter += 1;
                counter %= 100;
            }
        });
    } else {
        let _ = rt.run(fg);
    }

    Ok(())
}