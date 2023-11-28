#![allow(clippy::new_ret_no_self)]
mod clock_recovery_mm;
pub use clock_recovery_mm::ClockRecoveryMm;

mod decoder;
pub use decoder::Decoder;

mod fft_shift;
pub use fft_shift::FftShift;

mod iq_delay;
pub use iq_delay::IqDelay;

mod keep_1_in_n;
pub use keep_1_in_n::Keep1InN;

mod mac;
pub use mac::Mac;

mod modulator;
pub use modulator::modulator;

#[cfg(target_arch = "wasm32")]
pub mod wasm_rx;

use futuresdr::anyhow::{bail, Result};

pub fn channel_to_freq(chan: u32) -> Result<f64> {
    if (11..=26).contains(&chan) {
        Ok((2400.0 + 5.0 * (chan as f64 - 10.0)) * 1e6)
    } else {
        bail!("wrong channel {chan}");
    }
}

pub fn parse_channel(s: &str) -> Result<f64, String> {
    let channel: u32 = s
        .parse()
        .map_err(|_| format!("`{s}` isn't a ZigBee channel number"))?;

    channel_to_freq(channel).map_err(|_| format!("`{s}` isn't a ZigBee channel number"))
}
