use clap::Parser;
use futuredsp::firdes::lowpass;
use futuredsp::windows::hamming;
use futuresdr::anyhow::Result;
use futuresdr::async_io::Timer;
use futuresdr::blocks::seify::SinkBuilder;
use futuresdr::blocks::seify::SourceBuilder;
use futuresdr::blocks::NullSink;
use futuresdr::log::{debug, info};
use futuresdr::num_complex::Complex32;
use futuresdr::runtime::buffer::circular::Circular;
use futuresdr::runtime::Flowgraph;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::Runtime;
use lora::{PfbChannelizer, StreamDeinterleaver};
use seify::Device;
use seify::Direction::{Rx, Tx};
use std::time::Duration;

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    /// Soapy device Filter
    #[clap(long)]
    device_filter: Option<String>,
    /// Zigbee RX Gain
    #[clap(long, default_value_t = 50.0)]
    rx_gain: f64,
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
    /// Zigbee RX Frequency Offset
    #[clap(long, default_value_t = 0.0)]
    rx_freq_offset: f64,
    /// lora bandwidth
    #[clap(long, default_value_t = 125000)]
    bandwidth: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let rt = Runtime::new();
    let mut fg = Flowgraph::new();

    let filter = args.device_filter.unwrap_or_else(|| String::new());
    let is_soapy_dev = filter.clone().contains("driver=soapy");
    // println!("is_soapy_dev: {}", is_soapy_dev);
    let seify_dev = Device::from_args(&*filter).unwrap();

    seify_dev.set_sample_rate(Tx, 0, 125000.).unwrap();
    seify_dev.set_sample_rate(Rx, 0, 600000.).unwrap();

    if is_soapy_dev {
        info!("setting soapy frequencies");
        // else use specified center frequency and offset
        seify_dev
            .set_component_frequency(Tx, 0, "RF", args.center_freq)
            .unwrap();
        seify_dev
            .set_component_frequency(Tx, 0, "BB", args.tx_freq_offset)
            .unwrap();
    } else {
        // is aaronia device, no offset for TX and only one real center freq, tx center freq has to be set as center_freq+offset; also other component names
        info!("setting aaronia frequencies");
        seify_dev
            .set_component_frequency(Tx, 0, "RF", args.center_freq + args.tx_freq_offset)
            .unwrap();
        seify_dev
            .set_component_frequency(Rx, 0, "RF", args.center_freq)
            .unwrap();
        seify_dev
            .set_component_frequency(Rx, 0, "DEMOD", args.rx_freq_offset)
            .unwrap();
    }

    let mut sink = SinkBuilder::new()
        .driver(if is_soapy_dev {
            "soapy"
        } else {
            "aaronia_http"
        })
        .device(seify_dev.clone())
        .gain(args.tx_gain);
    // .dev_channels(vec![args.soapy_rx_channel]);
    let mut src = SourceBuilder::new()
        .driver(if is_soapy_dev {
            "soapy"
        } else {
            "aaronia_http"
        })
        .device(seify_dev)
        .gain(args.rx_gain);

    let sink = fg.add_block(sink.build().unwrap());
    let src = fg.add_block(src.build().unwrap());

    let deinterleaver = fg.add_block(StreamDeinterleaver::<Complex32>::new(3));
    fg.connect_stream(src, "out", deinterleaver, "in")?;
    let filter_coefs = lowpass(0.208333333, &hamming(24, false));
    let channelizer = fg.add_block(PfbChannelizer::new(3, &filter_coefs, 1.));
    fg.connect_stream(deinterleaver, "out0", channelizer, "in0")?;
    fg.connect_stream(deinterleaver, "out1", channelizer, "in1")?;
    fg.connect_stream(deinterleaver, "out2", channelizer, "in2")?;
    let null_sink21 = fg.add_block(NullSink::<Complex32>::new());
    // let null_sink22 = fg.add_block(NullSink::<Complex32>::new());
    let null_sink23 = fg.add_block(NullSink::<Complex32>::new());
    fg.connect_stream(channelizer, "out0", null_sink21, "in")?;
    // fg.connect_stream(channelizer, "out1", null_sink22, "in")?;
    fg.connect_stream(channelizer, "out2", null_sink23, "in")?;
    // fg.connect_stream(deinterleaver, "out0", null_sink21, "in")?;
    // fg.connect_stream(deinterleaver, "out1", null_sink22, "in")?;
    // fg.connect_stream(deinterleaver, "out2", null_sink23, "in")?;

    // let null_sink2 = fg.add_block(NullSink::<Complex32>::new());
    // fg.connect_stream(deinterleaver, "out0", null_sink2, "in")?;
    // let null_sink3 = fg.add_block(NullSink::<Complex32>::new());
    // fg.connect_stream(deinterleaver, "out2", null_sink3, "in")?;
    // fg.connect_stream(channelizer, "out1", sink, "in")?;
    fg.connect_stream_with_type(
        channelizer,
        "out1",
        sink,
        "in",
        Circular::with_size(2 * 4 * 8192 * 4),
    )?;

    let _ = rt.run(fg);

    Ok(())
}
