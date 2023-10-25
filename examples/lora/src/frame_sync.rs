use futuresdr::anyhow::Result;
use futuresdr::async_trait::async_trait;
use std::f32::consts::PI;
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

// #[derive(Debug)]
// enum Ev {
//     ModeManual(f32),
//     ModeAutomaticFreeSpace,
//     ModeAutomaticFlatEarthTwoRay,
//     ModeAutomaticCurvedEarthTwoRay,
//     ModeAutomaticNineRay,
//     Value(f32, f32, f32, f32, f32, f32),
//     ScalingCoeff(f32),
// }

#[derive(Debug, Copy, Clone)]
enum DecoderState {
    DETECT,
    SYNC,
    SFO_COMPENSATION,
    STOP,
}
#[derive(Debug, Copy, Clone)]
enum SyncState {
    NET_ID1 = 0,
    NET_ID2 = 1,
    DOWNCHIRP1 = 2,
    DOWNCHIRP2 = 3,
    QUARTER_DOWN = 4,
    SYNCED(usize),
}

pub struct FrameSync {
    m_state: DecoderState, //< Current state of the synchronization
    m_center_freq: u32,    //< RF center frequency
    m_bw: u32,             //< Bandwidth
    // m_samp_rate: u32,               //< Sampling rate
    m_sf: u8, //< Spreading factor
    // m_cr: u8,                       //< Coding rate
    // m_pay_len: u32,                 //< payload length
    // m_has_crc: u8,                  //< CRC presence
    // m_invalid_header: u8,           //< invalid header checksum
    m_impl_head: bool,      //< use implicit header mode
    m_os_factor: u8,        //< oversampling factor
    m_sync_words: Vec<u16>, //< vector containing the two sync words (network identifiers)
    // m_ldro: bool,                        //< use of low datarate optimisation mode
    m_n_up_req: u8, //< number of consecutive upchirps required to trigger a detection

    m_number_of_bins: usize,     //< Number of bins in each lora Symbol
    m_samples_per_symbol: usize, //< Number of samples received per lora symbols
    m_symb_numb: usize,          //<number of payload lora symbols
    m_received_head: bool, //< indicate that the header has be decoded and received by this block
    // m_noise_est: f64,            //< estimate of the noise
    in_down: Vec<Complex32>,     //< downsampled input
    m_downchirp: Vec<Complex32>, //< Reference downchirp
    m_upchirp: Vec<Complex32>,   //< Reference upchirp

    frame_cnt: usize,      //< Number of frame received
    symbol_cnt: SyncState, //< Number of symbols already received
    bin_idx: i32,          //< value of previous lora symbol
    // bin_idx_new: i32, //< value of newly demodulated symbol
    m_preamb_len: usize,        //< Number of consecutive upchirps in preamble
    additional_upchirps: usize, //< indicate the number of additional upchirps found in preamble (in addition to the minimum required to trigger a detection)

    cx_in: Vec<Complex32>,  //<input of the FFT
    cx_out: Vec<Complex32>, //<output of the FFT

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
    k_hat: i32,            //< integer part of CFO+STO
    preamb_up_vals: Vec<i32>, //< value of the preamble upchirps

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

fn build_upchirp(id: u32, sf: u8) -> Vec<Complex32> {
    let n = (1 << sf) as i32;
    let n_fold: i32 = n - id as i32;
    let mut chirp = vec![Complex32::from(0.); (1 << sf) as usize];
    for i in 0..n as i32 {
        if n < n_fold {
            chirp[n] = Complex32::new(1.0, 0.0)
                * (Complex32::new(0.0, 1.0)
                    * Complex32::new(
                        (2.0 * PI * (i * i / (2 * n) + (id as f32 / n as f32 - 0.5) * i)),
                        0.0,
                    ))
                .exp();
        } else {
            chirp[n] = Complex32::new(1.0, 0.0)
                * (Complex32::new(0.0, 1.0)
                    * Complex32::new(
                        (2.0 * PI * (i * i / (2 * n) + (id as f32 / n as f32 - 1.5) * i)),
                        0.0,
                    ))
                .exp();
        }
    }
    chirp
}

fn volk_32fc_conjugate_32fc(a_vector: &Vec<Complex32>) -> Vec<Complex32> {
    let mut b_vector = vec![Complex32::from(0.); a_vector.len()];
    for i in ..a_vector.len() {
        let tmp: Complex32 = a_vector[i];
        b_vector[i] = tmp.conj();
    }
    b_vector
}

fn build_ref_chirps(sf: u8) -> (Vec<Complex32>, Vec<Complex32>) {
    let n: f64 = (1 << sf) as f64;
    let upchirp = build_upchirp(0, sf);
    let downchirp = volk_32fc_conjugate_32fc(&upchirp);
    (upchirp, downchirp)
}

impl FrameSync {
    pub fn new(
        center_freq: u32,
        bandwidth: u32,
        sf: u8,
        impl_head: bool,
        sync_word: Vec<u16>,
        os_factor: u8,
        preamble_len: Option<u16>,
    ) -> Block {
        if preamble_len < 5 {
            warn!("Preamble length should be greater than 5!"); // TODO
        }
        let preamble_len_tmp = preamble_len.unwrap_or(8);
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
                .add_input("noise_est", Self::noise_est_handler)
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

                m_n_up_req: preamble_len_tmp - 3, //< number of consecutive upchirps required to trigger a detection
                up_symb_to_use: preamble_len_tmp - 4, //< number of upchirp symbols to use for CFO and STO frac estimation

                m_sto_frac: 0.0, //< fractional part of CFO

                m_impl_head: impl_head, //< use implicit header mode

                m_number_of_bins: m_number_of_bins_tmp, //< Number of bins in each lora Symbol
                m_samples_per_symbol: m_samples_per_symbol_tmp, //< Number of samples received per lora symbols
                additional_symbol_samp: vec![0; 2 * m_samples_per_symbol_tmp], //< save the value of the last 1.25 downchirp as it might contain the first payload symbol
                m_upchirp: m_upchirp_tmp,                                      //< Reference upchirp
                m_downchirp: m_downchirp_tmp, //< Reference downchirp
                preamble_upchirps: vec![0; m_number_of_bins_tmp], //<vector containing the preamble upchirps
                preamble_raw_up: vec![0; m_number_of_bins_tmp], //<vector containing the upsampled preamble upchirps without any synchronization
                CFO_frac_correc: vec![0; m_number_of_bins_tmp], //< cfo frac correction vector
                CFO_SFO_frac_correc: vec![0; m_number_of_bins_tmp], //< correction vector accounting for cfo and sfo
                symb_corr: vec![0; m_number_of_bins_tmp], //< symbol with CFO frac corrected
                in_down: vec![0; m_number_of_bins_tmp],   //< downsampled input
                preamble_raw: vec![0; m_number_of_bins_tmp * preamble_len_tmp], //<vector containing the preamble upchirps without any synchronization
                net_id_samp: vec![0; (m_samples_per_symbol_tmp as f32 * 2.5) as usize], //< vector of the oversampled network identifier samples

                bin_idx: 0,                     //< value of previous lora symbol
                symbol_cnt: SyncState::NET_ID2, //< Number of symbols already received  // TODO
                k_hat: 0,                       //< integer part of CFO+STO
                preamb_up_vals: vec![0; preamble_len_tmp - 3], //< value of the preamble upchirps
                frame_cnt: 0,                   //< Number of frame received

                cx_in: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //<input of the FFT
                cx_out: vec![Complex32::new(0., 0.); m_number_of_bins_tmp], //<output of the FFT

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
}
//       /**
//           *  \brief  Estimate the value of fractional part of the CFO using RCTSL and correct the received preamble accordingly
//           *  \param  samples
//           *          The pointer to the preamble beginning.(We might want to avoid the
//           *          first symbol since it might be incomplete)
//           */
//       float estimate_CFO_frac(gr_complex *samples);
//       /**
//           *  \brief  (not used) Estimate the value of fractional part of the CFO using Berniers algorithm and correct the received preamble accordingly
//           *  \param  samples
//           *          The pointer to the preamble beginning.(We might want to avoid the
//           *          first symbol since it might be incomplete)
//           */
//       float estimate_CFO_frac_Bernier(gr_complex *samples);
//       /**
//           *  \brief  Estimate the value of fractional part of the STO from m_consec_up and returns the estimated value
//           *
//           **/
//       float estimate_STO_frac();
//       /**
//           *  \brief  Recover the lora symbol value using argmax of the dechirped symbol FFT. Returns -1 in case of an fft window containing no energy to handle noiseless simulations.
//           *
//           *  \param  samples
//           *          The pointer to the symbol beginning.
//           *  \param  ref_chirp
//           *          The reference chirp to use to dechirp the lora symbol.
//           */
//       uint32_t get_symbol_val(const gr_complex *samples, gr_complex *ref_chirp);
//
//
//       /**
//           *  \brief  Determine the energy of a symbol.
//           *
//           *  \param  samples
//           *          The complex symbol to analyse.
//           *          length
//           *          The number of LoRa symbols used for the estimation
//           */
//       float determine_energy(const gr_complex *samples, int length);
//
//       /**
//          *   \brief  Handle the reception of the explicit header information, received from the header_decoder block
//          */
//       void frame_info_handler(pmt::pmt_t frame_info);
//
//       /**
//           *  \brief  Handles reception of the noise estimate
//           */
//       void noise_est_handler(pmt::pmt_t noise_est);
//       /**
//           *  \brief  Set new SF received in a tag (used for CRAN)
//           */
//       void set_sf(int sf);
//
//       float determine_snr(const gr_complex *samples);
//
//     public:
//       frame_sync_impl(uint32_t center_freq, uint32_t bandwidth, uint8_t sf, bool impl_head, std::vector<uint16_t> sync_word, uint8_t os_factor, uint16_t preamb_len);
//       ~frame_sync_impl();
//
//       // Where all the action really happens
//       void forecast(int noutput_items, gr_vector_int &ninput_items_required);
//
//       int general_work(int noutput_items,
//                        gr_vector_int &ninput_items,
//                        gr_vector_const_void_star &input_items,
//                        gr_vector_void_star &output_items);
//     };
//
//   } // namespace lora_sdr
// } // namespace gr
