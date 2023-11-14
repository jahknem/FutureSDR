use clap::Parser;
use futuresdr::futures::channel::mpsc;
use futuresdr::futures::StreamExt;

use futuresdr::anyhow::Result;
use futuresdr::blocks::seify::SinkBuilder;
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
use lora::{AddCrc, GrayDemap, HammingEnc, Header, Interleaver, Modulate, Whitening};
use seify::Device;
use seify::Direction::Tx;

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    /// RX Antenna
    #[clap(long)]
    tx_antenna: Option<String>,
    /// Soapy device Filter
    #[clap(long)]
    device_filter: Option<String>,
    /// Zigbee RX Gain
    #[clap(long, default_value_t = 50.0)]
    tx_gain: f64,
    /// Zigbee Sample Rate
    #[clap(long, default_value_t = 4e6)]
    sample_rate: f64,
    /// Zigbee TX/RX Center Frequency
    #[clap(long, default_value_t = 2.45e9)]
    center_freq: f64,
    /// Zigbee RX Frequency Offset
    #[clap(long, default_value_t = 0.0)]
    tx_freq_offset: f64,
    /// Soapy RX Channel
    #[clap(long, default_value_t = 0)]
    soapy_tx_channel: usize,
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
        .set_sample_rate(Tx, args.soapy_tx_channel, args.sample_rate)
        .unwrap();

    if is_soapy_dev {
        info!("setting soapy frequencies");
        // else use specified center frequency and offset
        seify_dev
            .set_component_frequency(Tx, args.soapy_tx_channel, "RF", args.center_freq)
            .unwrap();
        seify_dev
            .set_component_frequency(Tx, args.soapy_tx_channel, "BB", args.tx_freq_offset)
            .unwrap();
    } else {
        // is aaronia device, no offset for TX and only one real center freq, tx center freq has to be set as center_freq+offset; also other component names
        info!("setting aaronia frequencies");
        seify_dev
            .set_component_frequency(Tx, args.soapy_tx_channel, "RF", args.center_freq)
            .unwrap();
        seify_dev
            .set_component_frequency(Tx, args.soapy_tx_channel, "DEMOD", args.tx_freq_offset)
            .unwrap();
    }

    let mut sink = SinkBuilder::new()
        .driver(if is_soapy_dev {
            "soapy"
        } else {
            "aaronia_http"
        })
        .device(seify_dev)
        .gain(args.tx_gain);
    // .dev_channels(vec![args.soapy_rx_channel]);

    if let Some(a) = args.tx_antenna {
        sink = sink.antenna(a);
    }

    let sink = fg.add_block(sink.build().unwrap());

    let sf = 7;
    let samp_rate = 250000;
    let impl_head = false;
    let has_crc = true;
    let frame_period = 1000;
    let cr = 1;
    let bw = 125000;

    let whitening = fg.add_block(Whitening::new(false, false));
    let header = fg.add_block(Header::new(impl_head, has_crc, cr));
    fg.connect_stream(whitening, "out", header, "in")?;
    let add_crc = fg.add_block(AddCrc::new(has_crc));
    fg.connect_stream(header, "out", add_crc, "in")?;
    let hamming_enc = fg.add_block(HammingEnc::new(cr, sf));
    fg.connect_stream(add_crc, "out", hamming_enc, "in")?;
    let interleaver = fg.add_block(Interleaver::new(cr as usize, sf, 0, bw));
    fg.connect_stream(hamming_enc, "out", interleaver, "in")?;
    let gray_demap = fg.add_block(GrayDemap::new(sf));
    fg.connect_stream(interleaver, "out", gray_demap, "in")?;
    let modulate = fg.add_block(Modulate::new(
        sf,
        samp_rate,
        bw,
        vec![8, 16],
        20 * (1 << sf) * samp_rate / bw,
        Some(8),
    ));
    fg.connect_stream(gray_demap, "out", modulate, "in")?;
    fg.connect_stream(modulate, "out", sink, "in")?;

    let (_fg, _handle) = rt.start_sync(fg);
    loop {}

    Ok(())
}
