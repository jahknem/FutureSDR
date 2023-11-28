use clap::Parser;

use futuresdr::anyhow::Result;
use futuresdr::blocks::seify::SourceBuilder;

use futuresdr::blocks::NullSink;
use futuresdr::log::info;

use futuresdr::runtime::buffer::circular::Circular;
use futuresdr::runtime::Flowgraph;

use futuresdr::runtime::Runtime;
use rustfft::num_complex::Complex32;

use lora::{
    optfir, CrcVerif, Deinterleaver, Dewhitening, FftDemod, FrameSync, GrayMapping, HammingDec,
    HeaderDecoder, MmseResampler, PfbChannelizer, StreamDeinterleaver,
};
use seify::Device;
use seify::Direction::Rx;

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    /// RX Antenna
    #[clap(long)]
    rx_antenna: Option<String>,
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
    /// lora spreading factor
    #[clap(long, default_value_t = 7)]
    spreading_factor: usize,
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
    println!("is_soapy_dev: {}", is_soapy_dev);
    let seify_dev = Device::from_args(&*filter).unwrap();

    seify_dev
        // .set_sample_rate(Rx, args.soapy_rx_channel, args.sample_rate)
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

    let soft_decoding: bool = false;

    let deinterleaver = fg.add_block(StreamDeinterleaver::<Complex32>::new(3));
    fg.connect_stream(src, "out", deinterleaver, "in")?;
    // let filter_coefs = lowpass(0.208333333, &hamming(24, false));
    let transition_bw = (200_000. - 125_000.) / 200_000.;
    // let transition_bw = 0.2;
    let filter_coefs = optfir::low_pass(
        1.,
        3,
        0.5 - transition_bw / 2.,
        0.5 + transition_bw / 2.,
        0.1,
        100.,
        None,
    );
    let filter_coefs: Vec<f32> = filter_coefs.iter().map(|&x| x as f32).collect();
    println!("filter taps: {:?}", filter_coefs);
    let channelizer = fg.add_block(PfbChannelizer::new(3, &filter_coefs, 1.));
    fg.connect_stream(deinterleaver, "out0", channelizer, "in0")?;
    fg.connect_stream(deinterleaver, "out1", channelizer, "in1")?;
    fg.connect_stream(deinterleaver, "out2", channelizer, "in2")?;
    let null_sink21 = fg.add_block(NullSink::<Complex32>::new());
    let null_sink23 = fg.add_block(NullSink::<Complex32>::new());
    fg.connect_stream(channelizer, "out1", null_sink21, "in")?;
    fg.connect_stream(channelizer, "out0", null_sink23, "in")?;
    // fg.connect_stream(deinterleaver, "out0", null_sink21, "in")?;
    // fg.connect_stream(deinterleaver, "out1", null_sink23, "in")?;
    let resampler = fg.add_block(MmseResampler::<Complex32>::new(0., 1.6));
    fg.connect_stream(channelizer, "out2", resampler, "in")?;

    let frame_sync = fg.add_block(FrameSync::new(
        (args.center_freq + args.rx_freq_offset) as u32,
        // 868300,
        args.bandwidth as u32,
        args.spreading_factor,
        false,
        vec![0x12], // TODO 18?
        1,
        None,
    ));
    // fg.connect_stream_with_type(
    //     src,
    //     "out",
    //     frame_sync,
    //     "in",
    //     Circular::with_size(2 * 4 * 8192 * 4),
    // )?;
    fg.connect_stream_with_type(
        resampler,
        "out",
        frame_sync,
        "in",
        Circular::with_size(2 * 4 * 8192 * 4),
    )?;
    let null_sink2 = fg.add_block(NullSink::<f32>::new());
    fg.connect_stream(frame_sync, "log_out", null_sink2, "in")?;
    let fft_demod = fg.add_block(FftDemod::new(soft_decoding, true, args.spreading_factor));
    fg.connect_stream(frame_sync, "out", fft_demod, "in")?;
    let gray_mapping = fg.add_block(GrayMapping::new(soft_decoding));
    fg.connect_stream(fft_demod, "out", gray_mapping, "in")?;
    let deinterleaver = fg.add_block(Deinterleaver::new(soft_decoding));
    fg.connect_stream(gray_mapping, "out", deinterleaver, "in")?;
    let hamming_dec = fg.add_block(HammingDec::new(soft_decoding));
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

    let _ = rt.run(fg);

    Ok(())
}
