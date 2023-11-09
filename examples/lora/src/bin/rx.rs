use clap::Parser;
use futuresdr::futures::channel::mpsc;
use futuresdr::futures::StreamExt;

use futuresdr::anyhow::Result;
use futuresdr::blocks::seify::SourceBuilder;
use futuresdr::blocks::Apply;
use futuresdr::blocks::Combine;
use futuresdr::blocks::Fft;
use futuresdr::blocks::MessagePipe;
use futuresdr::blocks::NullSink;
use futuresdr::log::info;
use futuresdr::macros::connect;
use futuresdr::num_complex::Complex32;
use futuresdr::runtime::Flowgraph;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::Runtime;
use lora::utilities::*;
use lora::{
    CrcVerif, Deinterleaver, Dewhitening, FftDemod, FrameSync, GrayMapping, HammingDec,
    HeaderDecoder,
};
use seify::Device;
use seify::Direction::{Rx, Tx};

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    /// RX Antenna
    #[clap(long)]
    rx_antenna: Option<String>,
    /// TX Antenna
    #[clap(long)]
    tx_antenna: Option<String>,
    /// Soapy device Filter
    #[clap(long)]
    device_filter: Option<String>,
    /// Zigbee RX Gain
    #[clap(long, default_value_t = 50.0)]
    rx_gain: f64,
    /// Zigbee Sample Rate
    #[clap(long, default_value_t = 4e6)]
    sample_rate: f64,
    /// Zigbee TX/RX Center Frequency
    #[clap(long, default_value_t = 2.45e9)]
    center_freq: f64,
    /// Zigbee RX Frequency Offset
    #[clap(long, default_value_t = 0.0)]
    rx_freq_offset: f64,
    /// Soapy RX Channel
    #[clap(long, default_value_t = 0)]
    soapy_rx_channel: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let rt = Runtime::new();
    let mut fg = Flowgraph::new();

    let filter = args.device_filter.unwrap_or_else(|| "".to_string());
    let is_soapy_dev = filter.clone().contains("driver=soapy");
    println!("is_soapy_dev: {}", is_soapy_dev);
    let seify_dev = Device::from_args(&*filter).unwrap();

    seify_dev
        .set_sample_rate(Rx, args.soapy_rx_channel, args.sample_rate)
        .unwrap();

    if is_soapy_dev {
        info!("setting soapy frequencies");
        // else use specified center frequency and offset
        seify_dev
            .set_component_frequency(Rx, args.soapy_rx_channel, "RF", args.center_freq)
            .unwrap();
        seify_dev
            .set_component_frequency(Rx, args.soapy_rx_channel, "BB", args.rx_freq_offset)
            .unwrap();
    } else {
        // is aaronia device, no offset for TX and only one real center freq, tx center freq has to be set as center_freq+offset; also other component names
        info!("setting aaronia frequencies");
        seify_dev
            .set_component_frequency(Rx, args.soapy_rx_channel, "RF", args.center_freq)
            .unwrap();
        seify_dev
            .set_component_frequency(Rx, args.soapy_rx_channel, "DEMOD", args.rx_freq_offset)
            .unwrap();
    }

    let mut src = SourceBuilder::new()
        .driver(if is_soapy_dev {
            "soapy"
        } else {
            "aaronia_http"
        })
        .device(seify_dev)
        .gain(args.rx_gain);
    // .dev_channels(vec![args.soapy_rx_channel]);

    if let Some(a) = args.rx_antenna {
        src = src.antenna(a);
    }

    let src = fg.add_block(src.build().unwrap());

    let frame_sync = fg.add_block(FrameSync::new(
        868100000,
        125000,
        7,
        false,
        vec![0x12], // TODO 18?
        1,
        None,
    ));
    fg.connect_stream(src, "out", frame_sync, "in")?;
    let null_sink2 = fg.add_block(NullSink::<f32>::new());
    fg.connect_stream(frame_sync, "log_out", null_sink2, "in")?;
    let fft_demod = fg.add_block(FftDemod::new(true, true, 7));
    fg.connect_stream(frame_sync, "out", fft_demod, "in")?;
    let gray_mapping = fg.add_block(GrayMapping::new(true));
    fg.connect_stream(fft_demod, "out", gray_mapping, "in")?;
    let deinterleaver = fg.add_block(Deinterleaver::new(true));
    fg.connect_stream(gray_mapping, "out", deinterleaver, "in")?;
    let hamming_dec = fg.add_block(HammingDec::new(true));
    fg.connect_stream(deinterleaver, "out", hamming_dec, "in")?;
    let header_decoder = fg.add_block(HeaderDecoder::new(false, 1, 11, true, false, true));
    fg.connect_stream(hamming_dec, "out", header_decoder, "in")?;
    let dewhitening = fg.add_block(Dewhitening::new());
    fg.connect_stream(header_decoder, "out", dewhitening, "in")?;
    let crc_verif = fg.add_block(CrcVerif::new(true, false));
    fg.connect_stream(dewhitening, "out", crc_verif, "in")?;
    let null_sink3 = fg.add_block(NullSink::<bool>::new());
    fg.connect_stream(crc_verif, "out1", null_sink3, "in")?;

    fg.connect_message(header_decoder, "frame_info", frame_sync, "frame_info")?;

    let null_sink = fg.add_block(NullSink::<u8>::new());
    fg.connect_stream(crc_verif, "out", null_sink, "in")?;

    let (_fg, _handle) = rt.start_sync(fg);
    loop {}

    Ok(())
}
