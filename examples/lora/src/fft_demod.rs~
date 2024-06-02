use futuresdr::anyhow::Result;
use futuresdr::macros::async_trait;
use std::collections::HashMap;

// use futuresdr::futures::FutureExt;
use futuresdr::log::warn;

use futuresdr::num_complex::{Complex32, Complex64};
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
use futuresdr::runtime::{Block, ItemTag};

use scilib::math::bessel;

use crate::utilities::*;

use rustfft::{FftDirection, FftPlanner};

pub struct FftDemod {
    m_sf: usize,           //< Spreading factor
    m_cr: usize,           //< Coding rate
    m_soft_decoding: bool, //< Hard/Soft decoding
    max_log_approx: bool,  //< use Max-log approximation in LLR formula
    // m_new_frame: bool,     //< To be notify when receive a new frame to estimate SNR
    m_ldro: bool,       //< use low datarate optimisation
    m_symb_numb: usize, //< number of symbols in the frame
    // m_symb_cnt: usize, //< number of symbol already output in current frame
    // m_Ps_est: f64,               // Signal Power estimation updated at each rx symbol
    // m_Pn_est: f64,               // Signal Power estimation updated at each rx symbo
    m_samples_per_symbol: usize, //< Number of samples received per lora symbols
    // CFOint: i32,                 //< integer part of the CFO

    // variable used to perform the FFT demodulation
    m_upchirp: Vec<Complex32>,   //< Reference upchirp
    m_downchirp: Vec<Complex32>, //< Reference downchirp
    // m_dechirped: Vec<Complex32>, //< Dechirped symbol
    // m_fft: Vec<Complex32>, //< Result of the FFT
    output: Vec<u16>, //< Stores the value to be outputted once a full bloc has been received
    llrs_block: Vec<[LLR; MAX_SF]>, //< Stores the LLRs to be outputted once a full bloc has been received
    is_header: bool,                //< Indicate that the first block hasn't been fully received
                                    // block_size: usize,         //< The number of lora symbol in one block

                                    // #ifdef GRLORA_MEASUREMENTS
                                    // std::ofstream energy_file;
                                    // #endif
                                    // #ifdef GRLORA_DEBUG
                                    // std::ofstream idx_file;
                                    // #endif
                                    // #ifdef GRLORA_SNR_MEASUREMENTS_SAVE
                                    // std::ofstream SNRestim_file;
                                    // #endif
                                    // #ifdef GRLORA_BESSEL_MEASUREMENTS_SAVE
                                    // std::ofstream bessel_file;
                                    // #endif

                                    //   /**
                                    //    *  \brief  Reset the block variables when a new lora packet needs to be decoded.
                                    //    */
                                    // //   void new_frame_handler(int cfo_int);
                                    // //  // TODO unused?
}

impl FftDemod {
    pub fn new(soft_decoding: bool, max_log_approx: bool, sf_initial: usize) -> Block {
        let mut fs = Self {
            m_sf: 0,
            m_cr: 1, // TODO never initialized in cpp code (would set the value implicitly to 0), but set to 1 in python example
            m_soft_decoding: soft_decoding,
            max_log_approx,
            // m_new_frame: true,
            // m_symb_cnt: 0,
            // m_Ps_est: 0.,
            // m_Pn_est: 0.,
            m_samples_per_symbol: 0,
            // CFOint: _,
            m_upchirp: vec![],
            m_downchirp: vec![],
            // m_dechirped: vec![],
            // m_fft: vec![],
            output: vec![],
            llrs_block: vec![],
            is_header: false,
            m_ldro: false,
            m_symb_numb: 0,
            // block_size: _,
        };
        fs.set_sf(sf_initial); //accept any new sf
        let mut sio = StreamIoBuilder::new().add_input::<Complex32>("in");
        if soft_decoding {
            sio = sio.add_output::<[LLR; MAX_SF]>("out")
        } else {
            sio = sio.add_output::<u16>("out")
        }
        Block::new(
            BlockMetaBuilder::new("FftDemod").build(),
            sio.build(),
            MessageIoBuilder::new().build(),
            fs,
        )
        // #ifdef GRLORA_MEASUREMENTS
        //             int num = 0;  // check next file name to use
        //             while (1) {
        //                 std::ifstream infile("../../matlab/measurements/energy" + std::to_string(num) + ".txt");
        //                 if (!infile.good())
        //                     break;
        //                 num++;
        //             }
        //             energy_file.open("../../matlab/measurements/energy" + std::to_string(num) + ".txt", std::ios::out | std::ios::trunc);
        // #endif
        // #ifdef GRLORA_DEBUG
        //             idx_file.open("../data/idx.txt", std::ios::out | std::ios::trunc);
        // #endif
        // #ifdef GRLORA_SNR_MEASUREMENTS_SAVE
        //             SNRestim_file.open("../data/SNR_estimation.txt", std::ios::out | std::ios::trunc); //std::ios::trunc);
        //             //SNRestim_file << "New exp" << std::endl;
        // #endif
        // #ifdef GRLORA_BESSEL_MEASUREMENTS_SAVE
        //             bessel_file.open("../data/BesselArg.txt", std::ios::out | std::ios::trunc);
        // #endif
    }

    //         void fft_demod_impl::forecast(int noutput_items, gr_vector_int &ninput_items_required) {
    //             ninput_items_required[0] = m_samples_per_symbol;
    //         }  // TODO check condition in work

    /// Set spreading factor and init vector sizes accordingly
    fn set_sf(&mut self, sf: usize) {
        //Set he new sf for the frame
        // info!("[fft_demod_impl.cc] new sf received {}", sf);
        self.m_sf = sf;
        self.m_samples_per_symbol = 1_usize << self.m_sf;
        // self.m_upchirp = vec![Complex32::new(0., 0.); self.m_samples_per_symbol];
        // self.m_downchirp = vec![Complex32::new(0., 0.); self.m_samples_per_symbol];

        // FFT demodulation preparations
        // self.m_fft = vec![Complex32::new(0., 0.); self.m_samples_per_symbol];
        // self.m_dechirped = vec![Complex32::new(0., 0.); self.m_samples_per_symbol];
    }

    ///Compute the FFT and fill the class attributes
    fn compute_fft_mag(&self, samples: &[Complex32]) -> Vec<f64> {
        // info!("samples: {:?}", samples);
        // info!("self.m_downchirp: {:?}", self.m_downchirp);
        // Multiply with ideal downchirp
        let mut m_dechirped = volk_32fc_x2_multiply_32fc(samples, &self.m_downchirp);
        assert!(m_dechirped.len() == self.m_samples_per_symbol);
        // info!("m_dechirped: {:?}", m_dechirped);
        // let mut cx_out: Vec<Complex32> = m_dechirped[0..self.m_samples_per_symbol].to_vec();
        // info!("cx_out: {:?}", cx_out);
        // do the FFT
        FftPlanner::new()
            .plan_fft(self.m_samples_per_symbol, FftDirection::Forward)
            .process(&mut m_dechirped);
        // info!("cx_out after fft: {:?}", cx_out);  // TODO here
        // Get magnitude squared
        let m_fft_mag_sq = volk_32fc_magnitude_squared_32f(&m_dechirped);
        // info!("m_fft_mag_sq: {:?}", m_fft_mag_sq);
        // let rec_en = m_fft_mag_sq.iter().fold(0., |acc, e| acc + e);
        m_fft_mag_sq.iter().map(|x| *x as f64).collect()
    }
    //
    /// Use in Hard-decoding
    /// Recover the lora symbol value using argmax of the dechirped symbol FFT.
    /// \param  samples
    ///         The pointer to the symbol beginning.
    fn get_symbol_val(&self, samples: &[Complex32]) -> u16 {
        let m_fft_mag_sq = self.compute_fft_mag(samples);
        // Return argmax
        let idx = argmax_float(&m_fft_mag_sq);
        // #ifdef GRLORA_MEASUREMENTS
        //             energy_file << std::fixed << std::setprecision(10) << m_fft_mag_sq[idx] << "," << m_fft_mag_sq[mod(idx - 1, m_samples_per_symbol)] << "," << m_fft_mag_sq[mod(idx + 1, m_samples_per_symbol)] << "," << rec_en << "," << std::endl;
        // #endif
        // std::cout<<"SNR est = "<<m_fft_mag_sq[idx]<<","<<rec_en<<","<<10*log10(m_fft_mag_sq[idx]/(rec_en-m_fft_mag_sq[idx]))<<std::endl;

        // #ifdef GRLORA_DEBUG
        //             idx_file << idx << ", ";
        // #endif
        idx.try_into().unwrap()
    }

    ///  Use in Soft-decoding
    /// Compute the Log-Likelihood Ratios of the SF nbr of bits
    fn get_llrs(&self, samples: &[Complex32]) -> [LLR; MAX_SF] {
        let mut m_fft_mag_sq = self.compute_fft_mag(samples);

        //             // Compute LLRs of the SF bits
        let mut lls: Vec<f64> = vec![0.; self.m_samples_per_symbol]; // 2**sf  Log-Likelihood
        let mut llrs: [LLR; MAX_SF] = [0.; MAX_SF]; //      Log-Likelihood Ratios

        // compute SNR estimate at each received symbol as SNR remains constant during 1 simulation run
        // Estimate signal power
        let symbol_idx = argmax_float(&m_fft_mag_sq);
        // info!("symbol_idx: {}", symbol_idx);

        // Estimate noise power
        let mut signal_energy: f64 = 0.;
        let mut noise_energy: f64 = 0.;

        let n_adjacent_bins = 1; // Put '0' for best accurate SNR estimation but if symbols energy splitted in 2 bins, put '1' for safety
        for (i, &frequency_bin_energy) in m_fft_mag_sq.iter().enumerate() {
            if ((i as isize - symbol_idx as isize).unsigned_abs() % (self.m_samples_per_symbol - 1))
                < 1 + n_adjacent_bins
            {
                signal_energy += frequency_bin_energy;
            } else {
                noise_energy += frequency_bin_energy;
            }
        }

        // If you want to use a normalized constant identical to all symbols within a frame, but it leads to same performance
        // Lowpass filter update
        //double p = 0.99; // proportion to keep
        //Ps_est = p*Ps_est + (1-p)*  signal_energy / m_samples_per_symbol;
        //Pn_est = p*Pn_est + (1-p)* noise_energy / (m_samples_per_symbol-1-2*n_adjacent_bins); // remove used bins for better estimation
        // Signal and noise power estimation for each received symbol
        let m_ps_est = signal_energy / self.m_samples_per_symbol as f64;
        let m_pn_est = noise_energy / (self.m_samples_per_symbol - 1 - 2 * n_adjacent_bins) as f64;

        // #ifdef GRLORA_SNR_MEASUREMENTS_SAVE
        //             SNRestim_file << std::setprecision(6) << m_Ps_est << "," << m_Pn_est << std::endl;
        // #endif
        /*static int num_frames = 0;
        if (m_new_frame) {
            Ps_frame = Ps_est;
            Pn_frame = Pn_est;
            m_new_frame = false; // will be set back to True by new_frame_handler()
            num_frames++;
            //if (num_frames % 100 == 0) std::cout << "-----> SNRdB estim: " << 10*std::log10(Ps_frame/Pn_frame) << std::endl;
        }*/

        // #ifdef GRLORA_BESSEL_MEASUREMENTS_SAVE
        //             for (uint32_t n = 0; n < m_samples_per_symbol; n++) {
        //                 bessel_file << std::setprecision(8) << std::sqrt(Ps_frame) / Pn_frame * std::sqrt(m_fft_mag_sq[n]) << ","  << Ps_frame << "," << Pn_frame << "," << m_fft_mag_sq[n] << std::endl;
        //             }
        // #endif
        let _snr_db_estimate = 10. * (m_ps_est / m_pn_est).log10();
        // info!("SNR {}", SNRdB_estimate);
        //  Normalize fft_mag to 1 to avoid Bessel overflow
        m_fft_mag_sq = m_fft_mag_sq
            .iter()
            .map(|x| x * self.m_samples_per_symbol as f64)
            .collect();
        // upgrade to avoid for loop
        // Normalized |Y[n]| * sqrt(N) => |Y[n]|² * N (depends on kiss FFT library)
        //m_fft_mag_sq[i] /= Ps_frame; // // Normalize to avoid Bessel overflow (does not change the performances)
        //
        let mut clipping = false;
        for n in 0..self.m_samples_per_symbol {
            let bessel_arg = m_ps_est.sqrt() / m_pn_est * m_fft_mag_sq[n].sqrt();
            // Manage overflow of Bessel function
            // 713 ~ log(std::numeric_limits<LLR>::max())
            // if bessel_arg < 713. {
            if bessel_arg < 100. {
                // TODO original limit produces NaNs
                // info!("bessel_arg: {}", bessel_arg);
                let tmp = bessel::i_nu(0., Complex64::new(bessel_arg, 0.));
                // info!("tmp: {}", tmp);
                assert!(tmp.im == 0.);
                lls[n] = tmp.re; // compute Bessel safely  // TODO correct construction of complex number?
            } else {
                //std::cerr << RED << "Log-Likelihood clipping :-( SNR: " << SNRdB_estimate << " |Y|: " << std::sqrt(m_fft_mag_sq[n]) << RESET << std::endl;
                //LLs[n] = std::numeric_limits<LLR>::max();  // clipping
                clipping = true;
                break;
            }
            if self.max_log_approx {
                lls[n] = lls[n].ln(); // Log-Likelihood
                                      //LLs[n] = m_fft_mag_sq[n]; // same performance with just |Y[n]| or |Y[n]|²
            }
        }
        // change to max-log formula with only |Y[n]|² to avoid overflows, solve LLR computation incapacity in high SNR
        if clipping {
            lls.copy_from_slice(&m_fft_mag_sq);
        }

        // Log-Likelihood Ratio estimations
        if self.max_log_approx {
            for i in 0..self.m_sf {
                // sf bits => sf LLRs
                let mut max_x1: f64 = 0.;
                let mut max_x0: f64 = 0.; // X1 = set of symbols where i-th bit is '1'
                for (n, &ll) in lls.iter().enumerate().take(self.m_samples_per_symbol) {
                    // for all symbols n : 0 --> 2^sf
                    // LoRa: shift by -1 and use reduce rate if first block (header)
                    let mut s: usize = my_modulo(n as isize - 1, 1 << self.m_sf)
                        / if self.is_header || self.m_ldro { 4 } else { 1 };
                    s = s ^ (s >> 1); // Gray encoding formula               // Gray demap before (in this block)
                    if (s & (1 << i)) != 0 {
                        // if i-th bit of symbol n is '1'
                        if ll > max_x1 {
                            max_x1 = ll
                        }
                    } else {
                        // if i-th bit of symbol n is '0'
                        if ll > max_x0 {
                            max_x0 = ll
                        }
                    }
                }
                llrs[self.m_sf - 1 - i] = max_x1 - max_x0; // [MSB ... ... LSB]
            }
        } else {
            // Without max-log approximation of the LLR estimation
            for i in 0..self.m_sf {
                let mut sum_x1: f64 = 0.;
                let mut sum_x0: f64 = 0.; // X1 = set of symbols where i-th bit is '1'
                for (n, &ll) in lls.iter().enumerate().take(self.m_samples_per_symbol) {
                    // for all symbols n : 0 --> 2^sf
                    let mut s: usize = ((n - 1) % (1 << self.m_sf))
                        / if self.is_header || self.m_ldro { 4 } else { 1 };
                    s = s ^ (s >> 1); // Gray demap
                    if (s & (1 << i)) != 0 {
                        sum_x1 += ll;
                    }
                    // Likelihood
                    else {
                        sum_x0 += ll;
                    }
                }
                llrs[self.m_sf - 1 - i] = sum_x1.ln() - sum_x0.ln();
                // [MSB ... ... LSB]
            }
        }

        // #ifdef GRLORA_LLR_MEASUREMENTS_SAVE
        //             // Save Log-Likelihood and LLR for debug
        //             std::ofstream LL_file, LLR_file;
        //             LL_file.open("../data/fft_LL.txt", std::ios::out | std::ios::trunc);
        //             LLR_file.open("../data/LLR.txt", std::ios::out | std::ios::trunc);
        //
        //             for (uint32_t n = 0; n < m_samples_per_symbol; n++)
        //                 LL_file << std::fixed << std::setprecision(10) << m_fft_mag_sq[n] << "," << LLs[n] << std::endl;
        //             LL_file.close();
        //             for (uint32_t i = 0; i < m_sf; i++) LLR_file << std::fixed << std::setprecision(10) << LLRs[i] << std::endl;
        //             LLR_file.close();
        // #endif

        llrs
    }

    // /// Handles the reception of the coding rate received by the header_decoder block.
    // fn header_cr_handler(&mut self, cr: Pmt) {
    //     if let Pmt::U64(cr_val) = cr {
    //         self.m_cr = cr_val;
    //     } else {
    //         panic!("received invalid Pmr in FftDemod::header_cr_handler");
    //     }
    // }  // TODO unused?
}

#[async_trait]
impl Kernel for FftDemod {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _m: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input = sio.input(0).slice::<Complex32>();
        let mut nitems_to_process = input.len();

        let mut items_to_output: usize = 0;
        let mut items_to_consume: usize = 0; //< Number of items to consume after each iteration of the general_work functio

        let mut tag_tmp: Option<HashMap<String, Pmt>> =
            sio.input(0).tags().iter().find_map(|x| match x {
                ItemTag {
                    index: _,
                    tag: Tag::NamedAny(n, val),
                } => {
                    if n == "frame_info" {
                        match (**val).downcast_ref().unwrap() {
                            Pmt::MapStrPmt(map) => Some(map.clone()),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            });

        if let Some(ref tag) = tag_tmp {
            self.is_header = if let Pmt::Bool(tmp) = tag.get("is_header").unwrap() {
                *tmp
            } else {
                panic!()
            };
            // info!(
            //     "FftDemod: received new tag: {}",
            //     if self.is_header { "header" } else { "body" }
            // );
            if self.is_header
            // new frame beginning
            {
                let cfo_int = if let Pmt::F32(tmp) = tag.get("cfo_int").unwrap() {
                    *tmp as isize
                } else {
                    panic!()
                };
                let cfo_frac = if let Pmt::F64(tmp) = tag.get("cfo_frac").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                let sf = if let Pmt::Usize(tmp) = tag.get("sf").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                if sf != self.m_sf {
                    self.set_sf(sf);
                }
                //create downchirp taking CFO_int into account
                self.m_upchirp =
                    build_upchirp(my_modulo(cfo_int, self.m_samples_per_symbol), self.m_sf, 1);
                self.m_downchirp = volk_32fc_conjugate_32fc(&self.m_upchirp);
                // adapt the downchirp to the cfo_frac of the frame
                let tmp: Vec<Complex64> = (0..self.m_samples_per_symbol)
                    .map(|x| {
                        Complex64::from_polar(
                            1.,
                            -2. * std::f64::consts::PI * cfo_frac
                                / self.m_samples_per_symbol as f64
                                * x as f64,
                        )
                    })
                    .collect();
                let mut m_downchirp_tmp: Vec<Complex64> = self
                    .m_downchirp
                    .iter()
                    .map(|x| Complex64::new(x.re as f64, x.im as f64))
                    .collect();
                m_downchirp_tmp = volk_32fc_x2_multiply_32fc(&m_downchirp_tmp, &tmp);
                self.m_downchirp = m_downchirp_tmp
                    .iter()
                    .map(|x| Complex32::new(x.re as f32, x.im as f32))
                    .collect();
                if self.m_soft_decoding {
                    self.llrs_block.clear(); // TODO not cleared in original code
                } else {
                    self.output.clear();
                }
            } else {
                self.m_cr = if let Pmt::Usize(tmp) = tag.get("cr").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                self.m_ldro = if let Pmt::Bool(tmp) = tag.get("ldro").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
                self.m_symb_numb = if let Pmt::Usize(tmp) = tag.get("symb_numb").unwrap() {
                    *tmp
                } else {
                    panic!()
                };
            }
        }

        let block_size = 4 + if self.is_header { 4 } else { self.m_cr };

        while self.output.len() < block_size  // only consume more if not currently waiting for space in out buffer
            && self.llrs_block.len() < block_size
            && nitems_to_process >= self.m_samples_per_symbol
        {
            // TODO changed to loop as preceding block produces burst of samples and then stalls, leading to scheduler calling this work function only once.
            // info!(
            //     "self.LLRs_block.len(): {}/{}",
            //     self.LLRs_block.len(),
            //     block_size
            // );
            let input_tmp =
                &input[items_to_consume..(items_to_consume + self.m_samples_per_symbol)];
            // info!("input_tmp: {:?}", input_tmp);
            if self.m_soft_decoding {
                self.llrs_block.push(self.get_llrs(input_tmp)); // Store 'sf' LLRs
            } else {
                // Hard decoding
                // shift by -1 and use reduce rate if first block (header)
                self.output.push(
                    my_modulo(self.get_symbol_val(input_tmp) as isize - 1, 1 << self.m_sf) as u16
                        / if self.is_header || self.m_ldro { 4 } else { 1 },
                );
            }
            items_to_consume += self.m_samples_per_symbol;
            if let Some(tag) = tag_tmp {
                sio.output(0).add_tag(
                    0,
                    Tag::NamedAny("frame_info".to_string(), Box::new(Pmt::MapStrPmt(tag))),
                );
                tag_tmp = None;
            }
            nitems_to_process = input.len() - items_to_consume;
            // self.m_symb_cnt += 1;  // TODO noop
            // if self.m_symb_cnt == self.m_symb_numb {
            //     // std::cout<<"fft_demod_impl.cc end of frame\n";
            //     // set_sf(0);
            //     self.m_symb_cnt = 0;
            // }
        }
        // else if nitems_to_process < self.m_samples_per_symbol {
        //     warn!("FftDemod: not enough samples for one symbol in input buffer, waiting for more samples.")
        // }

        if !self.m_soft_decoding && self.output.len() == block_size {
            let out_buf = sio.output(0).slice::<u16>();
            if out_buf.len() >= block_size {
                out_buf[0..block_size].copy_from_slice(&self.output);
                self.output.clear();
                items_to_output = block_size
            } else {
                warn!("FftDemod: not enough space in output buffer.");
            }
        } else if self.m_soft_decoding && self.llrs_block.len() == block_size {
            let out_buf = sio.output(0).slice::<[LLR; MAX_SF]>();
            if out_buf.len() >= block_size {
                out_buf[0..block_size].copy_from_slice(&self.llrs_block);
                self.llrs_block.clear();
                items_to_output = block_size
            } else {
                warn!("FftDemod: not enough space in output buffer.");
            }
        } // else nothing to output

        if items_to_consume > 0 {
            // info!("FftDemod: consuming {} samples", items_to_consume);
            sio.input(0).consume(items_to_consume);
        }
        if items_to_output > 0 {
            // info!("FftDemod: producing {} samples", items_to_output);
            sio.output(0).produce(items_to_output);
        }
        Ok(())
    }
}
