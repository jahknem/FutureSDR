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
use seify::Device;
use seify::Direction::{Rx, Tx};

use lora::frame_sync::FrameSync;

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
        args.center_freq as u32,
        args.sample_rate as u32,
        8,
        false,
        vec![0, 0],
        1,
        None,
    ));
    fg.connect_stream(src, "out", frame_sync, "in")?;

    let null_sink = fg.add_block(NullSink::<Complex32>::new());
    fg.connect_stream(frame_sync, "out", null_sink, "in")?;

    let (_fg, _handle) = rt.start_sync(fg);
    loop {}

    Ok(())
}
