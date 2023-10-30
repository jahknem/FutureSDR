use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use std::cmp::Eq;
use std::f32::consts::PI;
use std::mem;
// use futuresdr::futures::FutureExt;
use crate::frame_sync::DecoderState::DETECT;
use futuresdr::log::warn;
use futuresdr::macros::message_handler;
use futuresdr::num_complex::Complex32;
use futuresdr::runtime::Block;
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::Pmt;
use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::Tag;
use futuresdr::runtime::WorkIo;

use crate::utilities::*;

use rustfft::num_traits::Signed;
use rustfft::{FftDirection, FftPlanner};

// impl Copy for usize {}
// impl Copy for u8 {}
// impl Copy for u16 {}
// impl Copy for u32 {}
// impl Copy for i32 {}
// impl Copy for f32 {}
// impl Copy for f64 {}
// impl Copy for bool {}

#[derive(Debug, Copy, Clone)]
enum DecoderState {
    DETECT,
    SYNC,
    SFO_COMPENSATION,
    STOP,
}
#[repr(usize)]
#[derive(Debug, Copy, Clone, PartialEq)]
enum SyncState {
    NET_ID1 = 0,
    NET_ID2 = 1,
    DOWNCHIRP1 = 2,
    DOWNCHIRP2 = 3,
    QUARTER_DOWN = 4,
    SYNCED(usize),
}
impl From<usize> for SyncState {
    fn from(orig: usize) -> Self {
        match orig {
            0_usize => SyncState::NET_ID1,
            1_usize => SyncState::NET_ID2,
            2_usize => SyncState::DOWNCHIRP1,
            3_usize => SyncState::DOWNCHIRP2,
            4_usize => SyncState::QUARTER_DOWN,
            _ => {
                warn!("implicit conversion from usize to SyncState::SYNCED(usize)");
                SyncState::SYNCED(orig)
            }
        }
    }
}
impl Into<usize> for SyncState {
    fn into(self) -> usize {
        match self {
            SyncState::NET_ID1 => 0_usize,
            SyncState::NET_ID2 => 1_usize,
            SyncState::DOWNCHIRP1 => 2_usize,
            SyncState::DOWNCHIRP2 => 3_usize,
            SyncState::QUARTER_DOWN => 4_usize,
            SyncState::SYNCED(value) => value,
        }
    }
}

pub struct FrameSync {
    m_state: DecoderState, //< Current state of the synchronization
    m_center_freq: u32,    //< RF center frequency
    m_bw: u32,             //< Bandwidth
    // m_samp_rate: u32,               //< Sampling rate
    m_sf: usize, //< Spreading factor
    // m_cr: u8,                       //< Coding rate
    // m_pay_len: u32,                 //< payload length
    // m_has_crc: u8,                  //< CRC presence
    // m_invalid_header: u8,           //< invalid header checksum
    m_impl_head: bool,      //< use implicit header mode
    m_os_factor: usize,     //< oversampling factor
    m_sync_words: Vec<u16>, //< vector containing the two sync words (network identifiers)
    // m_ldro: bool,                        //< use of low datarate optimisation mode
    m_n_up_req: SyncState, //< number of consecutive upchirps required to trigger a detection

    m_number_of_bins: usize,     //< Number of bins in each lora Symbol
    m_samples_per_symbol: usize, //< Number of samples received per lora symbols
    m_symb_numb: usize,          //<number of payload lora symbols
    m_received_head: bool, //< indicate that the header has be decoded and received by this block
    // m_noise_est: f64,            //< estimate of the noise
    in_down: Vec<Complex32>,     //< downsampled input
    m_downchirp: Vec<Complex32>, //< Reference downchirp
    m_upchirp: Vec<Complex32>,   //< Reference upchirp

    frame_cnt: usize,       //< Number of frame received
    symbol_cnt: SyncState,  //< Number of symbols already received
    bin_idx: Option<usize>, //< value of previous lora symbol
    // bin_idx_new: i32, //< value of newly demodulated symbol
    m_preamb_len: usize,        //< Number of consecutive upchirps in preamble
    additional_upchirps: usize, //< indicate the number of additional upchirps found in preamble (in addition to the minimum required to trigger a detection)

    // cx_in: Vec<Complex32>,  //<input of the FFT
    // cx_out: Vec<Complex32>, //<output of the FFT
    items_to_consume: usize, //< Number of items to consume after each iteration of the general_work function

    // one_symbol_off: i32, //< indicate that we are offset by one symbol after the preamble  // TODO bool?
    additional_symbol_samp: Vec<Complex32>, //< save the value of the last 1.25 downchirp as it might contain the first payload symbol
    preamble_raw: Vec<Complex32>, //<vector containing the preamble upchirps without any synchronization
    preamble_raw_up: Vec<Complex32>, //<vector containing the upsampled preamble upchirps without any synchronization
    // downchirp_raw: Vec<Complex32>,    //< vetor containing the preamble downchirps without any synchronization
    preamble_upchirps: Vec<Complex32>, //<vector containing the preamble upchirps
    net_id_samp: Vec<Complex32>,       //< vector of the oversampled network identifier samples
    net_ids: Vec<i32>,                 //< values of the network identifiers received

    up_symb_to_use: usize, //< number of upchirp symbols to use for CFO and STO frac estimation
    k_hat: usize,          //< integer part of CFO+STO
    preamb_up_vals: Vec<usize>, //< value of the preamble upchirps

    m_cfo_frac: f32,         //< fractional part of CFO
    m_cfo_frac_bernier: f32, //< fractional part of CFO using Berniers algo
    // m_cfo_int: i32,                               //< integer part of CFO
    m_sto_frac: f32,                     //< fractional part of CFO
    sfo_hat: f32,                        //< estimated sampling frequency offset
    sfo_cum: f32,                        //< cumulation of the sfo
    cfo_frac_sto_frac_est: bool, //< indicate that the estimation of CFO_frac and STO_frac has been performed
    CFO_frac_correc: Vec<Complex32>, //< cfo frac correction vector
    CFO_SFO_frac_correc: Vec<Complex32>, //< correction vector accounting for cfo and sfo

    symb_corr: Vec<Complex32>, //< symbol with CFO frac corrected
    down_val: i32,             //< value of the preamble downchirps
    // net_id_off: i32,                    //< offset of the network identifier
    m_should_log: bool, //< indicate that the sync values should be logged
                        // off_by_one_id: f32, //< Indicate that the network identifiers where off by one and corrected (float used as saved in a float32 bin file)
}

impl FrameSync {
    pub fn new(
        center_freq: u32,
        bandwidth: u32,
        sf: usize,
        impl_head: bool,
        sync_word: Vec<u16>,
        os_factor: usize,
        preamble_len: Option<usize>,
    ) -> Block {
        let preamble_len_tmp = preamble_len.unwrap_or(8);
        if preamble_len_tmp < 5 {
            panic!("Preamble length should be greater than 5!"); // TODO
        }
        let sync_word_tmp: Vec<u16> = if sync_word.len() == 1 {
            let tmp = sync_word[0];
            vec![((tmp & 0xF0_u16) >> 4) << 3, (tmp & 0x0F_u16) << 3]
        } else {
            sync_word
        };
        let m_number_of_bins_tmp = 1_usize << sf;
        let m_samples_per_symbol_tmp = m_number_of_bins_tmp * os_factor as usize;
        let (m_upchirp_tmp, m_downchirp_tmp) = build_ref_chirps(sf); // vec![0; m_number_of_bins_tmp]

        Block::new(
            BlockMetaBuilder::new("FrameSync").build(),
            StreamIoBuilder::new()
                .add_input::<u8>("in")
                .add_output::<u8>("out")
                .build(),
            MessageIoBuilder::new()
                .add_input("frame_info", Self::frame_info_handler)
                // .add_input("noise_est", Self::noise_est_handler)
                .add_output("snr")
                .build(),
            FrameSync {
                m_state: DETECT,            //< Current state of the synchronization
                m_center_freq: center_freq, //< RF center frequency
                m_bw: bandwidth,            //< Bandwidth
                m_sf: sf,                   //< Spreading factor

                m_sync_words: sync_word_tmp, //< vector containing the two sync words (network identifiers)
                m_os_factor: os_factor,      //< oversampling factor

                m_preamb_len: preamble_len_tmp, //< Number of consecutive upchirps in preamble
                net_ids: vec![0_i32; 2],        //< values of the network identifiers received

                m_n_up_req: SyncState::from(preamble_len_tmp - 3), //< number of consecutive upchirps required to trigger a detection
                up_symb_to_use: preamble_len_tmp - 4, //< number of upchirp symbols to use for CFO and STO frac estimation

                m_sto_frac: 0.0, //< fractional part of CFO

                m_impl_head: impl_head, //< use implicit header mode

                m_number_of_bins: m_number_of_bins_tmp, //< Number of bins in each lora Symbol
                m_samples_per_symbol: m_samples_per_symbol_tmp, //< Number of samples received per lora symbols
                additional_symbol_samp: vec![Complex32::new(0., 0.); 2 * m_samples_per_symbol_tmp], //< save the value of the last 1.25 downchirp as it might contain the first payload symbol
                m_upchirp: m_upchirp_tmp,     //< Reference upchirp
                m_downchirp: m_downchirp_tmp, //< Reference downchirp
                preamble_upchirps: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //<vector containing the preamble upchirps
                preamble_raw_up: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //<vector containing the upsampled preamble upchirps without any synchronization
                CFO_frac_correc: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //< cfo frac correction vector
                CFO_SFO_frac_correc: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //< correction vector accounting for cfo and sfo
                symb_corr: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //< symbol with CFO frac corrected
                in_down: vec![Complex32::new(0., 0.); m_number_of_bins_tmp],   //< downsampled input
                preamble_raw: vec![Complex32::new(0., 0.); m_number_of_bins_tmp * preamble_len_tmp], //<vector containing the preamble upchirps without any synchronization
                net_id_samp: vec![
                    Complex32::new(0., 0.);
                    (m_samples_per_symbol_tmp as f32 * 2.5) as usize
                ], //< vector of the oversampled network identifier samples

                bin_idx: None,                  //< value of previous lora symbol
                symbol_cnt: SyncState::NET_ID2, //< Number of symbols already received  // TODO
                k_hat: 0,                       //< integer part of CFO+STO
                preamb_up_vals: vec![0; preamble_len_tmp - 3], //< value of the preamble upchirps
                frame_cnt: 0,                   //< Number of frame received

                // cx_in: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //<input of the FFT
                // cx_out: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //<output of the FFT

                // m_samp_rate: u32,               //< Sampling rate  // unused
                // m_cr: u8,                       //< Coding rate  // local to frame info handler
                // m_pay_len: u32,                 //< payload length  // local to frame info handler
                // m_has_crc: u8,                  //< CRC presence  // local to frame info handler
                // m_invalid_header: u8,           //< invalid header checksum  // local to frame info handler
                // m_ldro: bool,                        //< use of low datarate optimisation mode  // local to frame info handler
                m_symb_numb: 0,         //<number of payload lora symbols
                m_received_head: false, //< indicate that the header has be decoded and received by this block
                // m_noise_est: f64,            //< estimate of the noise  // local to noise est handler

                // bin_idx_new: i32, //< value of newly demodulated symbol  // local to work
                additional_upchirps: 0, //< indicate the number of additional upchirps found in preamble (in addition to the minimum required to trigger a detection)

                items_to_consume: 0, //< Number of items to consume after each iteration of the general_work function

                // one_symbol_off: i32, //< indicate that we are offset by one symbol after the preamble  // local to work
                // downchirp_raw: Vec<Complex32>,    //< vetor containing the preamble downchirps without any synchronization  // unused
                m_cfo_frac: 0.0,         //< fractional part of CFO
                m_cfo_frac_bernier: 0.0, //< fractional part of CFO using Berniers algo
                // m_cfo_int: i32,                               //< integer part of CFO  // local to work
                sfo_hat: 0.0,                 //< estimated sampling frequency offset
                sfo_cum: 0.0,                 //< cumulation of the sfo
                cfo_frac_sto_frac_est: false, //< indicate that the estimation of CFO_frac and STO_frac has been performed

                down_val: 0, //< value of the preamble downchirps
                // net_id_off: i32,                    //< offset of the network identifier  // local to work
                m_should_log: false, //< indicate that the sync values should be logged
                                     // off_by_one_id: f32  // local to work
            },
        )
    }

    fn my_roundf(number: f32) -> usize {
        if number > 0.0 {
            (number + 0.5) as usize
        } else {
            (number - 0.5).ceil() as usize
        }
    }

    // fn forecast(int noutput_items, gr_vector_int &ninput_items_required)
    //     {
    //         ninput_items_required[0] = (m_os_factor * (m_number_of_bins + 2));
    //     }

    fn estimate_CFO_frac(&mut self, samples: &Vec<Complex32>) -> f32 {
        // create longer downchirp
        let mut downchirp_aug: Vec<Complex32> =
            vec![Complex32::new(0., 0.); self.up_symb_to_use * self.m_number_of_bins];
        for i in 0_usize..self.up_symb_to_use {
            downchirp_aug[(i * self.m_number_of_bins)..((i + 1) * self.m_number_of_bins)]
                .copy_from_slice(&self.m_downchirp[0..self.m_number_of_bins]);
        }

        // Dechirping
        let dechirped: Vec<Complex32> = volk_32fc_x2_multiply_32fc(&samples, &downchirp_aug);
        // prepare FFT
        // zero padded
        // let mut cx_in_cfo: Vec<Complex32> = vec![Complex32::new(0., 0.), 2 * self.up_symb_to_use * self.m_number_of_bins];
        // cx_in_cfo[..(self.up_symb_to_use * self.m_number_of_bins)].copy_from_slice(dechirped.as_slice());
        let mut cx_out_cfo: Vec<Complex32> =
            vec![Complex32::new(0., 0.); 2 * self.up_symb_to_use * self.m_number_of_bins];
        cx_out_cfo[..(self.up_symb_to_use * self.m_number_of_bins)]
            .copy_from_slice(dechirped.as_slice());
        // do the FFT
        FftPlanner::new()
            .plan_fft(cx_out_cfo.len(), FftDirection::Forward)
            .process(&mut cx_out_cfo);
        // Get magnitude
        let fft_mag_sq: Vec<f32> = volk_32fc_magnitude_squared_32f(&cx_out_cfo);
        // get argmax here
        let k0: usize = argmax_float(&fft_mag_sq);

        // get three spectral lines
        let Y_1 = fft_mag_sq[(k0 - 1) % (2 * self.up_symb_to_use * self.m_number_of_bins)];
        let Y0 = fft_mag_sq[k0];
        let Y1 = fft_mag_sq[(k0 + 1) % (2 * self.up_symb_to_use * self.m_number_of_bins)];
        // set constant coeff
        let u = 64. * self.m_number_of_bins as f32 / 406.5506497; // from Cui yang (15)
        let v = u * 2.4674;
        // RCTSL
        let wa = (Y1 - Y_1) / (u * (Y1 + Y_1) + v * Y0);
        let ka = wa * self.m_number_of_bins as f32 / PI;
        let k_residual = ((k0 as f32 + ka) / 2. / self.up_symb_to_use as f32) % 1.; // TODO verify ordering od divisions is identical in C and rust
        let cfo_frac = k_residual - if k_residual > 0.5 { 1. } else { 0. };
        // Correct CFO frac in preamble
        let CFO_frac_correc_aug: Vec<Complex32> = (0_usize
            ..self.up_symb_to_use * self.m_number_of_bins)
            .map(|x| {
                Complex32::from_polar(
                    1.,
                    -2. * PI * (cfo_frac) / self.m_number_of_bins as f32 * x as f32,
                )
            })
            .collect();

        self.preamble_upchirps = volk_32fc_x2_multiply_32fc(&samples, &CFO_frac_correc_aug);

        cfo_frac
    }

    fn estimate_CFO_frac_Bernier(&mut self, samples: &Vec<Complex32>) -> f32 {
        let mut fft_val: Vec<Complex32> =
            vec![Complex32::new(0., 0.); self.up_symb_to_use * self.m_number_of_bins];
        let mut k0: Vec<usize> = vec![0; self.up_symb_to_use];
        let mut k0_mag: Vec<f32> = vec![0.; self.up_symb_to_use]; // TODO original type double
        for i in 0_usize..self.up_symb_to_use {
            // Dechirping
            let dechirped: Vec<Complex32> = volk_32fc_x2_multiply_32fc(&samples, &self.m_downchirp);
            let mut cx_out_cfo: Vec<Complex32> = dechirped;
            // do the FFT
            FftPlanner::new()
                .plan_fft(cx_out_cfo.len(), FftDirection::Forward)
                .process(&mut cx_out_cfo);
            let fft_mag_sq: Vec<f32> = volk_32fc_magnitude_squared_32f(&cx_out_cfo);
            fft_val[(i * self.m_number_of_bins)..((i + 1) * self.m_number_of_bins)]
                .copy_from_slice(&cx_out_cfo[0_usize..self.m_number_of_bins]);
            // Get magnitude
            // get argmax here
            k0[i] = argmax_float(&fft_mag_sq);

            k0_mag[i] = fft_mag_sq[k0[i]];
        }
        // get argmax
        let idx_max: usize = argmax_float(&k0_mag);
        let mut four_cum = Complex32::new(0., 0.);
        for i in 0_usize..(self.up_symb_to_use - 1) {
            four_cum += fft_val[idx_max + self.m_number_of_bins * i]
                * (fft_val[idx_max + self.m_number_of_bins * (i + 1)]).conj();
        }
        let cfo_frac = -four_cum.arg() / 2. / PI;
        // Correct CFO in preamble
        let CFO_frac_correc_aug: Vec<Complex32> = (0_usize
            ..(self.up_symb_to_use * self.m_number_of_bins))
            .map(|x| {
                Complex32::from_polar(
                    1.,
                    -2. * PI * cfo_frac / self.m_number_of_bins as f32 * x as f32,
                )
            })
            .collect();
        self.preamble_upchirps = volk_32fc_x2_multiply_32fc(&samples, &CFO_frac_correc_aug);
        cfo_frac
    }

    fn estimate_STO_frac(&self) -> f32 {
        // int k0;
        // double Y_1, Y0, Y1, u, v, ka, wa, k_residual;
        // float sto_frac = 0;

        // std::vector<gr_complex> dechirped(m_number_of_bins);
        // kiss_fft_cpx *cx_in_sto = new kiss_fft_cpx[2 * m_number_of_bins];
        // kiss_fft_cpx *cx_out_sto = new kiss_fft_cpx[2 * m_number_of_bins];

        // std::vector<float> fft_mag_sq(2 * m_number_of_bins);
        // for (size_t i = 0; i < 2 * m_number_of_bins; i++)
        // {
        //     fft_mag_sq[i] = 0;
        // }
        // kiss_fft_cfg cfg_sto = kiss_fft_alloc(2 * m_number_of_bins, 0, 0, 0);

        let mut fft_mag_sq: Vec<f32> = vec![0.; 2 * self.m_number_of_bins];
        for i in 0_usize..self.up_symb_to_use {
            // Dechirping
            let dechirped: Vec<Complex32> = volk_32fc_x2_multiply_32fc(
                &self.preamble_upchirps
                    [(self.m_number_of_bins * i)..(self.m_number_of_bins * (i + 1))],
                &self.m_downchirp,
            );

            let mut cx_out_sto: Vec<Complex32> =
                vec![Complex32::new(0., 0.); 2 * self.m_number_of_bins];
            cx_out_sto[..self.m_number_of_bins].copy_from_slice(&dechirped);
            // do the FFT
            FftPlanner::new()
                .plan_fft(cx_out_sto.len(), FftDirection::Forward)
                .process(&mut cx_out_sto);
            // Get magnitude

            fft_mag_sq = volk_32fc_magnitude_squared_32f(&cx_out_sto)
                .iter()
                .zip(fft_mag_sq.iter())
                .map(|(x, y)| x + y)
                .collect();
        }

        // get argmax here
        let k0 = argmax_float(&fft_mag_sq);

        // get three spectral lines
        let Y_1 = fft_mag_sq[(k0 - 1) % (2 * self.m_number_of_bins)] as f64;
        let Y0 = fft_mag_sq[k0] as f64;
        let Y1 = fft_mag_sq[(k0 + 1) % (2 * self.m_number_of_bins)] as f64;

        // set constant coeff
        let u = 64. * self.m_number_of_bins as f64 / 406.5506497; // from Cui yang (eq.15)
        let v = u * 2.4674;
        // RCTSL
        let wa = (Y1 - Y_1) / (u * (Y1 + Y_1) + v * Y0);
        let ka = wa * self.m_number_of_bins as f64 / std::f64::consts::PI;
        let k_residual = ((k0 as f64 + ka) / 2.) % (1.);
        let sto_frac = (k_residual - if k_residual > 0.5 { 1. } else { 0. }) as f32;

        sto_frac
    }

    fn get_symbol_val(samples: &[Complex32], ref_chirp: &[Complex32]) -> Option<usize> {
        // double sig_en = 0;
        // std::vector<float> fft_mag(m_number_of_bins);
        // volk::vector<gr_complex> dechirped(m_number_of_bins);

        // kiss_fft_cfg cfg = kiss_fft_alloc(m_number_of_bins, 0, 0, 0);

        // Multiply with ideal downchirp
        let dechirped = volk_32fc_x2_multiply_32fc(&samples, &ref_chirp);

        let mut cx_out: Vec<Complex32> = dechirped;
        // do the FFT
        FftPlanner::new()
            .plan_fft(cx_out.len(), FftDirection::Forward)
            .process(&mut cx_out);

        // Get magnitude
        let fft_mag = volk_32fc_magnitude_squared_32f(&cx_out);
        //     sig_en += fft_mag[i];
        // }
        let sig_en: f64 = fft_mag.iter().map(|x| *x as f64).fold(0., |acc, e| acc + e);
        // Return argmax here

        return if sig_en != 0. {
            Some(argmax_float(&fft_mag))
        } else {
            None
        };
    }

    fn determine_energy(&self, samples: &Vec<Complex32>, length: Option<usize>) -> f32 {
        let length_tmp = length.unwrap_or(1);
        let magsq_chirp = volk_32fc_magnitude_squared_32f(
            &samples[0_usize..(self.m_number_of_bins * length_tmp)],
        );
        let energy_chirp = magsq_chirp.iter().fold(0., |acc, e| acc + e);
        return energy_chirp / self.m_number_of_bins as f32 / length_tmp as f32;
    }

    fn determine_snr(&self, samples: &Vec<Complex32>) -> f32 {
        // double tot_en = 0;
        // std::vector<float> fft_mag(m_number_of_bins);
        // std::vector<gr_complex> dechirped(m_number_of_bins);

        // kiss_fft_cfg cfg = kiss_fft_alloc(m_number_of_bins, 0, 0, 0);

        // Multiply with ideal downchirp
        let dechirped = volk_32fc_x2_multiply_32fc(&samples, &self.m_downchirp);

        let mut cx_out: Vec<Complex32> = dechirped;
        // do the FFT
        FftPlanner::new()
            .plan_fft(cx_out.len(), FftDirection::Forward)
            .process(&mut cx_out);

        // Get magnitude
        let fft_mag = volk_32fc_magnitude_squared_32f(&cx_out);
        //     sig_en += fft_mag[i];
        // }
        let tot_en: f64 = fft_mag.iter().map(|x| *x as f64).fold(0., |acc, e| acc + e);
        // Return argmax here
        let max_idx = argmax_float(&fft_mag);
        let sig_en = fft_mag[max_idx] as f64;
        return (10. * (sig_en / (tot_en - sig_en)).log10()) as f32;
    }

    // TODO self.m_noise_est only referenced here, so a noop?
    // #[message_handler]
    // fn noise_est_handler(
    //     &mut self,
    //     _io: &mut WorkIo,
    //     mio: &mut MessageIo<Self>,
    //     _meta: &mut BlockMeta,
    //     p: Pmt,
    // ) -> Result<Pmt> {
    //     if let Pmt::F32(noise_est) = p {
    //         self.m_noise_est = p;
    //     } else {
    //         warn!("noise_est pmt was not an f32");
    //     }
    //     Ok(Pmt::Null)
    // }

    #[message_handler]
    fn frame_info_handler(
        &mut self,
        _io: &mut WorkIo,
        mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
        p: Pmt,
    ) -> Result<Pmt> {
        if let Pmt::MapStrPmt(mut frame_info) = p {
            // TODO
            // pmt::pmt_t
            let err = Pmt::String(String::from("error"));
            //
            let m_cr: usize = if let Pmt::Usize(temp) = frame_info.get("cr").unwrap_or(&err) {
                *temp
            } else {
                panic!("invalid cr")
            }; // TODO double
            let m_pay_len: usize =
                if let Pmt::Usize(temp) = frame_info.get("pay_len").unwrap_or(&err) {
                    *temp
                } else {
                    panic!("invalid pay_len")
                };
            let m_has_crc: usize = if let Pmt::Usize(temp) = frame_info.get("crc").unwrap_or(&err) {
                *temp
            } else {
                panic!("invalid m_has_crc")
            };
            // uint8_t
            let ldro_mode_tmp: ldro_mode =
                if let Pmt::Usize(temp) = frame_info.get("ldro_mode").unwrap_or(&err) {
                    (*temp).into()
                } else {
                    panic!("invalid ldro mode")
                };
            let m_invalid_header = frame_info.get("err").unwrap_or(&err);

            if *m_invalid_header == err {
                self.m_state = DETECT;
                self.symbol_cnt = SyncState::NET_ID2;
                self.k_hat = 0;
                self.m_sto_frac = 0.;
            } else {
                let m_ldro: ldro_mode = if ldro_mode_tmp == ldro_mode::AUTO {
                    if (1_usize << self.m_sf) as f32 * 1e3 / self.m_bw as f32 > LDRO_MAX_DURATION_MS
                    {
                        ldro_mode::ENABLE
                    } else {
                        ldro_mode::DISABLE
                    }
                } else {
                    ldro_mode_tmp
                };

                self.m_symb_numb = 8
                    + ((2 * m_pay_len - self.m_sf
                        + 2
                        + (!self.m_impl_head) as usize * 5
                        + m_has_crc * 4) as f64
                        / (self.m_sf - 2 * m_ldro as usize) as f64)
                        .ceil() as usize
                        * (4 + m_cr);
                self.m_received_head = true;
                frame_info.insert(String::from("is_header"), Pmt::Bool(false));
                frame_info.insert(String::from("symb_numb"), Pmt::Usize(self.m_symb_numb));
                frame_info.remove("ldro_mode");
                frame_info.insert(String::from("ldro"), Pmt::Bool(m_ldro as usize != 0));
                let frame_info_pmt = Pmt::MapStrPmt(frame_info);
                // TODO tag stream
                // add_item_tag(
                //     0,
                //     nitems_written(0),
                //     pmt::string_to_symbol("frame_info"),
                //     frame_info,
                // );
            }
        } else {
            warn!("noise_est pmt was not a Map/Dict");
        }
        Ok(Pmt::Null)
    }

    fn set_sf(&mut self, sf: usize) {
        self.m_sf = sf;
        self.m_number_of_bins = 1 << self.m_sf;
        self.m_samples_per_symbol = self.m_number_of_bins * self.m_os_factor;
        self.additional_symbol_samp
            .resize(2 * self.m_samples_per_symbol, Complex32::new(0., 0.));
        self.m_upchirp
            .resize(self.m_number_of_bins, Complex32::new(0., 0.));
        self.m_downchirp
            .resize(self.m_number_of_bins, Complex32::new(0., 0.));
        self.preamble_upchirps.resize(
            self.m_preamb_len * self.m_number_of_bins,
            Complex32::new(0., 0.),
        );
        self.preamble_raw_up.resize(
            (self.m_preamb_len + 3) * self.m_samples_per_symbol,
            Complex32::new(0., 0.),
        );
        self.CFO_frac_correc
            .resize(self.m_number_of_bins, Complex32::new(0., 0.));
        self.CFO_SFO_frac_correc
            .resize(self.m_number_of_bins, Complex32::new(0., 0.));
        self.symb_corr
            .resize(self.m_number_of_bins, Complex32::new(0., 0.));
        self.in_down
            .resize(self.m_number_of_bins, Complex32::new(0., 0.));
        self.preamble_raw.resize(
            self.m_preamb_len * self.m_number_of_bins,
            Complex32::new(0., 0.),
        );
        self.net_id_samp.resize(
            (self.m_samples_per_symbol as f32 * 2.5) as usize,
            Complex32::new(0., 0.),
        ); // we should be able to move up to one quarter of symbol in each direction
        let (upchirp_tmp, downchirp_tmp) = build_ref_chirps(self.m_sf);
        self.m_upchirp = upchirp_tmp;
        self.m_downchirp = downchirp_tmp;

        // self.cx_in = new kiss_fft_cpx[m_number_of_bins];
        // self.cx_out = new kiss_fft_cpx[m_number_of_bins];

        // Constrain the noutput_items argument passed to forecast and general_work.
        // set_output_multiple causes the scheduler to ensure that the noutput_items argument passed to forecast and general_work will be an integer multiple of
        // https://www.gnuradio.org/doc/doxygen/classgr_1_1block.html#a63d67fd758b70c6f2d7b7d4edcec53b3
        // set_output_multiple(m_number_of_bins);  // TODO
    }
}

#[async_trait]
impl Kernel for FrameSync {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        //
        //         int frame_sync_impl::general_work(int noutput_items,
        //                                           gr_vector_int &ninput_items,
        //                                           gr_vector_const_void_star &input_items,
        //                                           gr_vector_void_star &output_items)
        //         {
        //             const gr_complex *in = (const gr_complex *)input_items[0];
        //             gr_complex *out = (gr_complex *)output_items[0];
        let mut items_to_output: usize = 0;
        //
        let out = sio.output(0).slice::<Complex32>(); // TODO Complex32?
                                                      // check if there is enough space in the output buffer
        if out.len() < self.m_number_of_bins {
            return Ok(());
        }

        // float *sync_log_out = NULL;
        // if (output_items.size() == 2)  // TODO
        // {
        //     sync_log_out = (float *)output_items[1];
        //     m_should_log = true;
        // }
        // else
        //     m_should_log = false;
        let input = sio.input(0).slice::<Complex32>();
        let nitems_to_process = input.len();

        // let tags = sio.input(0).tags().iter().filter(|x| x.index < )
        //             std::vector<tag_t> tags;  // TODO
        //             get_tags_in_window(tags, 0, 0, ninput_items[0], pmt::string_to_symbol("new_frame"));
        //             if tags.size()
        //             {
        //                 if tags[0].offset != nitems_read(0)
        //                     nitems_to_process = tags[0].offset - nitems_read(0); // only use symbol until the next frame begin (SF might change)
        //
        //                 else
        //                 {
        //                     if tags.size() >= 2 {
        //                         nitems_to_process = tags[1].offset - tags[0].offset;
        //                     }
        //
        //                     pmt::pmt_t err = pmt::string_to_symbol("error");
        //
        //                     int sf = pmt::to_long(pmt::dict_ref(tags[0].value, pmt::string_to_symbol("sf"), err));
        //                     set_sf(sf);
        //
        //                     // std::cout<<"\nhamming_cr "<<tags[0].offset<<" - cr: "<<(int)m_cr<<"\n";
        //                 }
        //             }

        // downsampling
        let indexing_offset =
            self.m_os_factor / 2 - FrameSync::my_roundf(self.m_sto_frac * self.m_os_factor as f32);
        self.in_down = input
            [(indexing_offset..(indexing_offset + self.m_number_of_bins * self.m_os_factor))]
            .iter()
            .step_by(self.m_os_factor)
            .map(|x| *x)
            .collect();
        // for (uint32_t ii = 0; ii < m_number_of_bins; ii++)
        //     in_down[ii] = in[(int)(m_os_factor / 2 + m_os_factor * ii - my_roundf(m_sto_frac * m_os_factor))];

        match self.m_state {
            DecoderState::DETECT => {
                let bin_idx_new_opt = FrameSync::get_symbol_val(&self.in_down, &self.m_downchirp);

                let condition_failed = if let Some(bin_idx_new) = bin_idx_new_opt {
                    if ((((bin_idx_new as i32 - self.bin_idx.map(|x| x as i32).unwrap_or(-1))
                        .abs()
                        + 1)
                        % self.m_number_of_bins as i32)
                        - 1)
                    .abs()
                        <= 1
                    {
                        if let Some(bin_idx) = self.bin_idx {
                            if self.symbol_cnt == SyncState::NET_ID2 {
                                self.preamb_up_vals[0] = bin_idx;
                            }
                        }

                        self.preamb_up_vals[Into::<usize>::into(self.symbol_cnt)] = bin_idx_new;
                        let preamble_raw_idx_offset =
                            self.m_number_of_bins * Into::<usize>::into(self.symbol_cnt);
                        let count = self.m_number_of_bins * mem::size_of::<Complex32>();
                        self.preamble_raw
                            [preamble_raw_idx_offset..(preamble_raw_idx_offset + count)]
                            .copy_from_slice(&self.in_down[0..count]);
                        let preamble_raw_up_idx_offset =
                            self.m_samples_per_symbol * Into::<usize>::into(self.symbol_cnt);
                        let count = self.m_samples_per_symbol * mem::size_of::<Complex32>();
                        self.preamble_raw_up
                            [preamble_raw_up_idx_offset..(preamble_raw_up_idx_offset + count)]
                            .copy_from_slice(
                                &input[(self.m_os_factor / 2)..(self.m_os_factor / 2 + count)],
                            );

                        self.symbol_cnt =
                            From::<usize>::from((Into::<usize>::into(self.symbol_cnt) + 1_usize));
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
                if condition_failed {
                    let count = self.m_number_of_bins * mem::size_of::<Complex32>();
                    self.preamble_raw[0..count].copy_from_slice(&self.in_down[0..count]);
                    let count = self.m_samples_per_symbol * mem::size_of::<Complex32>();
                    self.preamble_raw_up[0..count].copy_from_slice(
                        &input[(self.m_os_factor / 2)..(self.m_os_factor / 2 + count)],
                    );

                    self.symbol_cnt = SyncState::NET_ID2;
                }
                self.bin_idx = bin_idx_new_opt;
                if self.symbol_cnt == self.m_n_up_req {
                    self.additional_upchirps = 0;
                    self.m_state = DecoderState::SYNC;
                    self.symbol_cnt = SyncState::NET_ID1;
                    self.cfo_frac_sto_frac_est = false;
                    self.k_hat = most_frequent(&self.preamb_up_vals);
                    let input_idx_offset = (0.75 * self.m_samples_per_symbol as f32
                        - self.k_hat as f32 * self.m_os_factor as f32)
                        as usize;
                    let count = (mem::size_of::<Complex32>() as f32 * 0.25) as usize
                        * self.m_samples_per_symbol;
                    self.net_id_samp[0..count]
                        .copy_from_slice(&input[input_idx_offset..(input_idx_offset + count)]);

                    // perform the coarse synchronization
                    self.items_to_consume = self.m_os_factor * (self.m_number_of_bins - self.k_hat);
                } else {
                    self.items_to_consume = self.m_samples_per_symbol;
                }
                items_to_output = 0;
            }
            DecoderState::SYNC => {
                //                 items_to_output = 0;
                //                 if (!cfo_frac_sto_frac_est)
                //                 {
                //                     m_cfo_frac = estimate_CFO_frac_Bernier(&preamble_raw[m_number_of_bins - k_hat]);
                //                     m_sto_frac = estimate_STO_frac();
                //                     // create correction vector
                //                     for (uint32_t n = 0; n < m_number_of_bins; n++)
                //                     {
                //                         CFO_frac_correc[n] = gr_expj(-2 * M_PI * m_cfo_frac / m_number_of_bins * n);
                //                     }
                //                     cfo_frac_sto_frac_est = true;
                //                 }
                //                 items_to_consume = m_samples_per_symbol;
                //                 // apply cfo correction
                //                 volk_32fc_x2_multiply_32fc(&symb_corr[0], &in_down[0], &CFO_frac_correc[0], m_number_of_bins);
                //
                //                 bin_idx = get_symbol_val(&symb_corr[0], &m_downchirp[0]);
                //                 switch (symbol_cnt)
                //                 {
                //                 case NET_ID1:
                //                 {
                //                     if (bin_idx == 0 || bin_idx == 1 || (uint32_t)bin_idx == m_number_of_bins - 1)
                //                     { // look for additional upchirps. Won't work if network identifier 1 equals 2^sf-1, 0 or 1!
                //                         memcpy(&net_id_samp[0], &in[(int)0.75 * m_samples_per_symbol], sizeof(gr_complex) * 0.25 * m_samples_per_symbol);
                //                         if (additional_upchirps >= 3)
                //                         {
                //                             std::rotate(preamble_raw_up.begin(), preamble_raw_up.begin() + m_samples_per_symbol, preamble_raw_up.end());
                //                             memcpy(&preamble_raw_up[m_samples_per_symbol * (m_n_up_req + 3)], &in[(int)(m_os_factor / 2) + k_hat * m_os_factor], m_samples_per_symbol * sizeof(gr_complex));
                //                         }
                //                         else
                //                         {
                //                             memcpy(&preamble_raw_up[m_samples_per_symbol * (m_n_up_req + additional_upchirps)], &in[(int)(m_os_factor / 2) + k_hat * m_os_factor], m_samples_per_symbol * sizeof(gr_complex));
                //                             additional_upchirps++;
                //                         }
                //                     }
                //                     else
                //                     { // network identifier 1 correct or off by one
                //                         symbol_cnt = NET_ID2;
                //                         memcpy(&net_id_samp[0.25 * m_samples_per_symbol], &in[0], sizeof(gr_complex) * m_samples_per_symbol);
                //                         net_ids[0] = bin_idx;
                //                     }
                //                     break;
                //                 }
                //                 case NET_ID2:
                //                 {
                //
                //                     symbol_cnt = DOWNCHIRP1;
                //                     memcpy(&net_id_samp[1.25 * m_samples_per_symbol], &in[0], sizeof(gr_complex) * (m_number_of_bins + 1) * m_os_factor);
                //                     net_ids[1] = bin_idx;
                //
                //                     break;
                //                 }
                //                 case DOWNCHIRP1:
                //                 {
                //                     memcpy(&net_id_samp[2.25 * m_samples_per_symbol], &in[0], sizeof(gr_complex) * 0.25 * m_samples_per_symbol);
                //                     symbol_cnt = DOWNCHIRP2;
                //                     break;
                //                 }
                //                 case DOWNCHIRP2:
                //                 {
                //                     down_val = get_symbol_val(&symb_corr[0], &m_upchirp[0]);
                //                     memcpy(&additional_symbol_samp[0], &in[0], sizeof(gr_complex) * m_samples_per_symbol);
                //                     symbol_cnt = QUARTER_DOWN;
                //                     break;
                //                 }
                //                 case QUARTER_DOWN:
                //                 {
                //                     memcpy(&additional_symbol_samp[m_samples_per_symbol], &in[0], sizeof(gr_complex) * m_samples_per_symbol);
                //                     if ((uint32_t)down_val < m_number_of_bins / 2)
                //                     {
                //                         m_cfo_int = floor(down_val / 2);
                //                     }
                //                     else
                //                     {
                //                         m_cfo_int = floor(double(down_val - (int)m_number_of_bins) / 2);
                //                     }
                //
                //                     // correct STOint and CFOint in the preamble upchirps
                //                     std::rotate(preamble_upchirps.begin(), preamble_upchirps.begin() + mod(m_cfo_int, m_number_of_bins), preamble_upchirps.end());
                //
                //                     std::vector<gr_complex> CFO_int_correc;
                //                     CFO_int_correc.resize((m_n_up_req + additional_upchirps) * m_number_of_bins);
                //                     for (uint32_t n = 0; n < (m_n_up_req + additional_upchirps) * m_number_of_bins; n++)
                //                     {
                //                         CFO_int_correc[n] = gr_expj(-2 * M_PI * (m_cfo_int) / m_number_of_bins * n);
                //                     }
                //
                //                     volk_32fc_x2_multiply_32fc(&preamble_upchirps[0], &preamble_upchirps[0], &CFO_int_correc[0], up_symb_to_use * m_number_of_bins);
                //
                //                     // correct SFO in the preamble upchirps
                //
                //                     sfo_hat = float((m_cfo_int + m_cfo_frac) * m_bw) / m_center_freq;
                //                     double clk_off = sfo_hat / m_number_of_bins;
                //                     double fs = m_bw;
                //                     double fs_p = m_bw * (1 - clk_off);
                //                     int N = m_number_of_bins;
                //                     std::vector<gr_complex> sfo_corr_vect;
                //                     sfo_corr_vect.resize((m_n_up_req + additional_upchirps) * m_number_of_bins, 0);
                //                     for (uint32_t n = 0; n < (m_n_up_req + additional_upchirps) * m_number_of_bins; n++)
                //                     {
                //                         sfo_corr_vect[n] = gr_expj(-2 * M_PI * (pow(mod(n, N), 2) / 2 / N * (m_bw / fs_p * m_bw / fs_p - m_bw / fs * m_bw / fs) + (std::floor((float)n / N) * (m_bw / fs_p * m_bw / fs_p - m_bw / fs_p) + m_bw / 2 * (1 / fs - 1 / fs_p)) * mod(n, N)));
                //                     }
                //
                //                     volk_32fc_x2_multiply_32fc(&preamble_upchirps[0], &preamble_upchirps[0], &sfo_corr_vect[0], up_symb_to_use * m_number_of_bins);
                //
                //                     float tmp_sto_frac = estimate_STO_frac(); // better estimation of sto_frac in the beginning of the upchirps
                //                     float diff_sto_frac = m_sto_frac - tmp_sto_frac;
                //
                //                     if (abs(diff_sto_frac) <= float(m_os_factor - 1) / m_os_factor) // avoid introducing off-by-one errors by estimating fine_sto=-0.499 , rough_sto=0.499
                //                         m_sto_frac = tmp_sto_frac;
                //
                //                     // get SNR estimate from preamble
                //                     // downsample preab_raw
                //                     std::vector<gr_complex> corr_preamb;
                //                     corr_preamb.resize((m_n_up_req + additional_upchirps) * m_number_of_bins, 0);
                //                     // apply sto correction
                //                     for (uint32_t i = 0; i < (m_n_up_req + additional_upchirps) * m_number_of_bins; i++)
                //                     {
                //                         corr_preamb[i] = preamble_raw_up[m_os_factor * (m_number_of_bins - k_hat + i) - int(my_roundf(m_os_factor * m_sto_frac))];
                //                     }
                //                     std::rotate(corr_preamb.begin(), corr_preamb.begin() + mod(m_cfo_int, m_number_of_bins), corr_preamb.end());
                //                     // apply cfo correction
                //                     volk_32fc_x2_multiply_32fc(&corr_preamb[0], &corr_preamb[0], &CFO_int_correc[0], (m_n_up_req + additional_upchirps) * m_number_of_bins);
                //                     for (int i = 0; i < (m_n_up_req + additional_upchirps); i++)
                //                     {
                //                         volk_32fc_x2_multiply_32fc(&corr_preamb[m_number_of_bins * i], &corr_preamb[m_number_of_bins * i], &CFO_frac_correc[0], m_number_of_bins);
                //                     }
                //
                //                     // //apply sfo correction
                //                     volk_32fc_x2_multiply_32fc(&corr_preamb[0], &corr_preamb[0], &sfo_corr_vect[0], (m_n_up_req + additional_upchirps) * m_number_of_bins);
                //
                //                     float snr_est = 0;
                //                     for (int i = 0; i < up_symb_to_use; i++)
                //                     {
                //                         snr_est += determine_snr(&corr_preamb[i * m_number_of_bins]);
                //                     }
                //                     snr_est /= up_symb_to_use;
                //
                //                     // update sto_frac to its value at the beginning of the net id
                //                     m_sto_frac += sfo_hat * m_preamb_len;
                //                     // ensure that m_sto_frac is in [-0.5,0.5]
                //                     if (abs(m_sto_frac) > 0.5)
                //                     {
                //                         m_sto_frac = m_sto_frac + (m_sto_frac > 0 ? -1 : 1);
                //                     }
                //                     // decim net id according to new sto_frac and sto int
                //                     std::vector<gr_complex> net_ids_samp_dec;
                //                     net_ids_samp_dec.resize(2 * m_number_of_bins, 0);
                //                     // start_off gives the offset in the net_id_samp vector required to be aligned in time (CFOint is equivalent to STOint since upchirp_val was forced to 0)
                //                     int start_off = (int)m_os_factor / 2 - (my_roundf(m_sto_frac * m_os_factor)) + m_os_factor * (.25 * m_number_of_bins + m_cfo_int);
                //                     for (uint32_t i = 0; i < m_number_of_bins * 2; i++)
                //                     {
                //                         net_ids_samp_dec[i] = net_id_samp[start_off + i * m_os_factor];
                //                     }
                //                     volk_32fc_x2_multiply_32fc(&net_ids_samp_dec[0], &net_ids_samp_dec[0], &CFO_int_correc[0], 2 * m_number_of_bins);
                //
                //                     // correct CFO_frac in the network ids
                //                     volk_32fc_x2_multiply_32fc(&net_ids_samp_dec[0], &net_ids_samp_dec[0], &CFO_frac_correc[0], m_number_of_bins);
                //                     volk_32fc_x2_multiply_32fc(&net_ids_samp_dec[m_number_of_bins], &net_ids_samp_dec[m_number_of_bins], &CFO_frac_correc[0], m_number_of_bins);
                //
                //                     int netid1 = get_symbol_val(&net_ids_samp_dec[0], &m_downchirp[0]);
                //                     int netid2 = get_symbol_val(&net_ids_samp_dec[m_number_of_bins], &m_downchirp[0]);
                //                     one_symbol_off = 0;
                //
                //                     if (abs(netid1 - (int32_t)m_sync_words[0]) > 2) // wrong id 1, (we allow an offset of 2)
                //                     {
                //
                //                         // check if we are in fact checking the second net ID and that the first one was considered as a preamble upchirp
                //                         if (abs(netid1 - (int32_t)m_sync_words[1]) <= 2)
                //                         {
                //                             net_id_off = netid1 - (int32_t)m_sync_words[1];
                //                             for (int i = m_preamb_len - 2; i < (m_n_up_req + additional_upchirps); i++)
                //                             {
                //                                 if (get_symbol_val(&corr_preamb[i * m_number_of_bins], &m_downchirp[0]) + net_id_off == m_sync_words[0]) // found the first netID
                //                                 {
                //                                     one_symbol_off = 1;
                //                                     if (net_id_off != 0 && abs(net_id_off) > 1)
                //                                         std::cout << RED << "[frame_sync_impl.cc] net id offset >1: " << net_id_off << RESET << std::endl;
                //                                     if (m_should_log)
                //                                         off_by_one_id = net_id_off != 0;
                //                                     items_to_consume = -m_os_factor * net_id_off;
                //                                     // the first symbol was mistaken for the end of the downchirp. we should correct and output it.
                //
                //                                     int start_off = (int)m_os_factor / 2 - my_roundf(m_sto_frac * m_os_factor) + m_os_factor * (0.25 * m_number_of_bins + m_cfo_int);
                //                                     for (int i = start_off; i < 1.25 * m_samples_per_symbol; i += m_os_factor)
                //                                     {
                //
                //                                         out[int((i - start_off) / m_os_factor)] = additional_symbol_samp[i];
                //                                     }
                //                                     items_to_output = m_number_of_bins;
                //                                     m_state = SFO_COMPENSATION;
                //                                     symbol_cnt = 1;
                //                                     frame_cnt++;
                //                                 }
                //                             }
                //                             if (!one_symbol_off)
                //                             {
                //                                 m_state = DETECT;
                //                                 symbol_cnt = 1;
                //                                 items_to_output = 0;
                //                                 k_hat = 0;
                //                                 m_sto_frac = 0;
                //                                 items_to_consume = 0;
                //                             }
                //                         }
                //                         else
                //                         {
                //                             m_state = DETECT;
                //                             symbol_cnt = 1;
                //                             items_to_output = 0;
                //                             k_hat = 0;
                //                             m_sto_frac = 0;
                //                             items_to_consume = 0;
                //                         }
                //                     }
                //                     else // net ID 1 valid
                //                     {
                //                         net_id_off = netid1 - (int32_t)m_sync_words[0];
                //                         if (mod(netid2 - net_id_off, m_number_of_bins) != (int32_t)m_sync_words[1]) // wrong id 2
                //                         {
                //                             m_state = DETECT;
                //                             symbol_cnt = 1;
                //                             items_to_output = 0;
                //                             k_hat = 0;
                //                             m_sto_frac = 0;
                //                             items_to_consume = 0;
                //                         }
                //                         else
                //                         {
                //                             if (net_id_off != 0 && abs(net_id_off) > 1)
                //                                 std::cout << RED << "[frame_sync_impl.cc] net id offset >1: " << net_id_off << RESET << std::endl;
                //                             if (m_should_log)
                //                                 off_by_one_id = net_id_off != 0;
                //                             items_to_consume = -m_os_factor * net_id_off;
                //                             m_state = SFO_COMPENSATION;
                //                             frame_cnt++;
                //                         }
                //                     }
                //                     if (m_state != DETECT)
                //                     {
                //                         // update sto_frac to its value at the payload beginning
                //                         m_sto_frac += sfo_hat * 4.25;
                //                         sfo_cum = ((m_sto_frac * m_os_factor) - my_roundf(m_sto_frac * m_os_factor)) / m_os_factor;
                //
                //                         pmt::pmt_t frame_info = pmt::make_dict();
                //                         frame_info = pmt::dict_add(frame_info, pmt::intern("is_header"), pmt::from_bool(true));
                //                         frame_info = pmt::dict_add(frame_info, pmt::intern("cfo_int"), pmt::mp((long)m_cfo_int));
                //                         frame_info = pmt::dict_add(frame_info, pmt::intern("cfo_frac"), pmt::mp((float)m_cfo_frac));
                //                         frame_info = pmt::dict_add(frame_info, pmt::intern("sf"), pmt::mp((long)m_sf));
                //
                //                         add_item_tag(0, nitems_written(0), pmt::string_to_symbol("frame_info"), frame_info);
                //
                //                         m_received_head = false;
                //                         items_to_consume += m_samples_per_symbol / 4 + m_os_factor * m_cfo_int;
                //                         symbol_cnt = one_symbol_off;
                //                         float snr_est2 = 0;
                //
                //                         if (m_should_log)
                //                         {
                //                             // estimate SNR
                //
                //                             for (int i = 0; i < up_symb_to_use; i++)
                //                             {
                //                                 snr_est2 += determine_snr(&preamble_upchirps[i * m_number_of_bins]);
                //                             }
                //                             snr_est2 /= up_symb_to_use;
                //                             float cfo_log = m_cfo_int + m_cfo_frac;
                //                             float sto_log = k_hat - m_cfo_int + m_sto_frac;
                //                             float srn_log = snr_est;
                //                             float sfo_log = sfo_hat;
                //
                //                             sync_log_out[0] = srn_log;
                //                             sync_log_out[1] = cfo_log;
                //                             sync_log_out[2] = sto_log;
                //                             sync_log_out[3] = sfo_log;
                //                             sync_log_out[4] = off_by_one_id;
                //                             produce(1, 5);
                //                         }
                // #ifdef PRINT_INFO
                //
                //                         std::cout << "[frame_sync_impl.cc] " << frame_cnt << " CFO estimate: " << m_cfo_int + m_cfo_frac << ", STO estimate: " << k_hat - m_cfo_int + m_sto_frac << " snr est: " << snr_est << std::endl;
                // #endif
                //                     }
                //                 }
                //                 }
                //
                //                 break;
            }
            DecoderState::SFO_COMPENSATION => {
                //                 // transmit only useful symbols (at least 8 symbol for PHY header)
                //
                //                 if (symbol_cnt < 8 || ((uint32_t)symbol_cnt < m_symb_numb && m_received_head))
                //                 {
                //                     // output downsampled signal (with no STO but with CFO)
                //                     memcpy(&out[0], &in_down[0], m_number_of_bins * sizeof(gr_complex));
                //                     items_to_consume = m_samples_per_symbol;
                //
                //                     //   update sfo evolution
                //                     if (abs(sfo_cum) > 1.0 / 2 / m_os_factor)
                //                     {
                //                         items_to_consume -= (-2 * signbit(sfo_cum) + 1);
                //                         sfo_cum -= (-2 * signbit(sfo_cum) + 1) * 1.0 / m_os_factor;
                //                     }
                //
                //                     sfo_cum += sfo_hat;
                //
                //                     items_to_output = m_number_of_bins;
                //                     symbol_cnt++;
                //                 }
                //                 else if (!m_received_head)
                //                 { // Wait for the header to be decoded
                //                     items_to_consume = 0;
                //                     items_to_output = 0;
                //                 }
                //                 else
                //                 {
                //                     m_state = DETECT;
                //                     symbol_cnt = 1;
                //                     items_to_consume = m_samples_per_symbol;
                //                     items_to_output = 0;
                //                     k_hat = 0;
                //                     m_sto_frac = 0;
                //                 }
                //                 break;
            }
            _ => {
                // std::cerr << "[LoRa sync] WARNING : No state! Shouldn't happen\n";
                // break;
            } //             }
              //             consume_each(items_to_consume);
              //             produce(0, items_to_output);
              //             return WORK_CALLED_PRODUCE;
        }
        //     } /* namespace lora_sdr */
        Ok(())
    }
} /* namespace gr */
