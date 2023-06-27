use std::collections::HashMap;
use std::thread::sleep;
use clap::Parser;
use std::time::Duration;
use async_process::Command;
use forky_tun::{self, Configuration};
// use futures::StreamExt;
// use futures::sink::SinkExt;
use std::net::Ipv4Addr;
use tokio;

use futuresdr::anyhow::Result;
use futuresdr::async_io;
use futuresdr::async_io::block_on;
use futuresdr::async_io::Timer;
use futuresdr::async_net::UdpSocket;
use futuresdr::blocks::Apply;
use futuresdr::blocks::Combine;
use futuresdr::blocks::Fft;
use futuresdr::blocks::FftDirection;
use futuresdr::blocks::FirBuilder;
use futuresdr::blocks::MessagePipe;
use futuresdr::blocks::Selector;
use futuresdr::blocks::SelectorDropPolicy as DropPolicy;
use futuresdr::futures::channel::mpsc;
use futuresdr::futures::StreamExt;
use futuresdr::log::info;
use futuresdr::log::warn;
use futuresdr::num_complex::Complex32;
use futuresdr::runtime::buffer::circular::Circular;
use futuresdr::runtime::Flowgraph;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::Runtime;

use multitrx::MessageSelector;

use multitrx::IPDSCPRewriter;
use multitrx::MetricsReporter;
use multitrx::TcpExchanger;
use multitrx::Complex32Serializer;
use multitrx::Complex32Deserializer;

use wlan::MAX_PAYLOAD_SIZE;
use wlan::fft_tag_propagation as wlan_fft_tag_propagation;
use wlan::Decoder as WlanDecoder;
use wlan::Delay as WlanDelay;
// use wlan::Encoder as WlanEncoder;
use multitrx::Encoder as WlanEncoder;
use wlan::FrameEqualizer as WlanFrameEqualizer;
use wlan::Mac as WlanMac;
use wlan::Mapper as WlanMapper;
use wlan::Mcs as WlanMcs;
use wlan::MovingAverage as WlanMovingAverage;
use wlan::Prefix as WlanPrefix;
use wlan::SyncLong as WlanSyncLong;
use wlan::SyncShort as WlanSyncShort;
use wlan::MAX_SYM;

use zigbee::modulator as zigbee_modulator;
use zigbee::IqDelay as ZigbeeIqDelay;
// use zigbee::Mac as ZigbeeMac;
use multitrx::ZigbeeMac;
use zigbee::ClockRecoveryMm as ZigbeeClockRecoveryMm;
use zigbee::Decoder as ZigbeeDecoder;


const PAD_FRONT: usize = 10000;
const PAD_TAIL: usize = 10000;

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    /// TCPExchanger remote server ip
    #[clap(long, value_parser, default_value = "0.0.0.0")]
    server_ip: String,
    /// TCPExchanger remote client ip
    #[clap(long, value_parser)]
    client_ip: String,
    /// UDP port to receive position updates
    #[clap(short, long, default_value_t = 1337)]
    local_udp_port: u32,
    /// UDP port to receive position updates
    #[clap(short, long, default_value_t = 1341)]
    model_selection_udp_port: u32,
    /// UDP port of channel emulator
    #[clap(short, long, default_value_t = 1338)]
    chanem_port: u32,
    /// Sample Rate
    #[clap(long, default_value_t = 200e6)]
    sample_rate: f64,
}


fn main() -> Result<()> {
    let args = Args::parse();
    println!("Configuration: {:?}", args);

    let mut fg = Flowgraph::new();

    //FIR
    let taps = [0.5f32, 0.5f32];

    let tcp_exchanger_to_uav = fg.add_block(TcpExchanger::new(args.client_ip, true));
    let tcp_exchanger_to_ground = fg.add_block(TcpExchanger::new(args.server_ip, false));
    let tcp_exchanger_to_uav = fg.add_block(TcpExchanger::new(args.local_ip.clone(), args.remote_ip.clone()));
    let iq_serializer_ag = fg.add_block(Complex32Serializer::new());
    let iq_deserializer_ag = fg.add_block(Complex32Deserializer::new());
    let fir_ag = fg.add_block(FirBuilder::new::<Complex32, Complex32, f32, _>(taps));
    let iq_serializer_ga = fg.add_block(Complex32Serializer::new());
    let iq_deserializer_ga = fg.add_block(Complex32Deserializer::new());
    let fir_ga = fg.add_block(FirBuilder::new::<Complex32, Complex32, f32, _>(taps));

    // ============================================
    // AG CHANNEL
    // ============================================
    fg.connect_stream(tcp_exchanger_to_uav, "out", iq_deserializer_ag, "in")?;
    fg.connect_stream(iq_deserializer_ag, "out", fir_ag, "in")?;
    fg.connect_stream(fir_ag, "out", iq_serializer_ag, "in")?;
    fg.connect_stream(iq_serializer_ag, "out", tcp_exchanger_to_ground, "in")?;

    // ============================================
    // GA CHANNEL
    // ============================================
    fg.connect_stream(tcp_exchanger_to_ground, "out", iq_deserializer_ga, "in")?;
    fg.connect_stream(iq_deserializer_ga, "out", fir_ga, "in")?;
    fg.connect_stream(fir_ga, "out", iq_serializer_ga, "in")?;
    fg.connect_stream(iq_serializer_ga, "out", tcp_exchanger_to_uav, "in")?;

    // ============================================
    // RUNTIME
    // ============================================
    let rt = Runtime::new();
    let (_fg, mut handle) = block_on(rt.start(fg));
    let mut input_handle = handle.clone();


    loop {
        sleep(Duration::from_secs(5));
    }

}
