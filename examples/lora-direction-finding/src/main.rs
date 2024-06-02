use futures::stream::StreamExt;

use futuresdr::anyhow::Result;
use futuresdr::runtime::{Runtime, Pmt};
use clap::Parser;

#[derive(Parser, Debug)]
#[clap(version)]
struct Args {
    /// RX Gain
    #[clap(long, default_value_t = 2e6)]
    sample_rate: f64,
    /// RX Gain
    #[clap(long, default_value_t = 50.0)]
    gain: f64,
    /// RX Frequency
    #[clap(long, default_value_t = 868.10e6)]
    frequency: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let soft_decoding: bool = false;

    let rt = Runtime::new();
    let (fg, mut receiver1, mut receiver2) = lora_direction_finding::build_flowgraph(
        args.sample_rate,
        args.frequency,
        args.gain,
    )?;

    rt.spawn_background(async move {
        while let Some(pmt) = receiver1.next().await {
            if let Pmt::Blob(bytes) = pmt {
                println!("received frame (ch. 1): {:02x?}", bytes);
            } else {
                println!("{:?}", pmt);
            }
        }
    });

    rt.spawn_background(async move {
        while let Some(pmt) = receiver2.next().await {
            if let Pmt::Blob(bytes) = pmt {
                println!("received frame (ch. 2): {:02x?}", bytes);
            } else {
                println!("{:?}", pmt);
            }
        }
    });

    let _ = rt.run(fg);

    Ok(())
}
