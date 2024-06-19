use futuresdr::anyhow::Result;
use futuresdr::macros::async_trait;
use futuresdr::runtime::BlockMeta;
use futuresdr::runtime::BlockMetaBuilder;
use futuresdr::runtime::Kernel;
use futuresdr::runtime::MessageIo;
use futuresdr::runtime::MessageIoBuilder;
use futuresdr::runtime::StreamIo;
use futuresdr::runtime::StreamIoBuilder;
use futuresdr::runtime::WorkIo;

use rustfft::num_complex::Complex;

pub struct PhaseDifference {
    fft_size: usize,
}

impl PhaseDifference {
    pub fn new(fft_size: usize) -> Block {
        Block::new(
            BlockMetaBuilder::new("PhaseDifference").build(),
            StreamIoBuilder::new().build(),
            MessageIoBuilder::new()
                .add_input("in1")
                .add_input("in2")
                .add_output("out")
                .build(),
            PhaseDifference {
                 fft_size 
            },
        )
    }
    #[message_handler]
    fn frame_info_handler(
        &mut self,
        _io: &mut WorkIo,
        _mio: &mut MessageIo<Self>,
        _meta: &mut BlockMeta,
        p: Pmt,
    ) -> Result<Pmt> {
        if let Pmt::MapStrPmt(mut frame_info) = p {
            let m_cr: usize = if let Pmt::Usize(temp) = frame_info.get("cr").unwrap() {
                *temp
            } else {
                panic!("invalid cr")
            };
            let m_pay_len: usize = if let Pmt::Usize(temp) = frame_info.get("pay_len").unwrap() {
                *temp
            } else {
                panic!("invalid pay_len")
            };
            let m_has_crc: bool = if let Pmt::Bool(temp) = frame_info.get("crc").unwrap() {
                *temp
            } else {
                panic!("invalid m_has_crc")
            };
            // uint8_t
            let ldro_mode_tmp: LdroMode =
                if let Pmt::Bool(temp) = frame_info.get("ldro_mode").unwrap() {
                    if *temp {
                        LdroMode::ENABLE
                    } else {
                        LdroMode::DISABLE
                    }
                } else {
                    panic!("invalid ldro_mode")
                };
            let m_invalid_header = if let Pmt::Bool(temp) = frame_info.get("err").unwrap() {
                *temp
            } else {
                panic!("invalid err flag")
            };

            // info!("FrameSync: received header info");

            if m_invalid_header {
                self.m_state = DecoderState::Detect;
                self.symbol_cnt = SyncState::NetId2;
                self.k_hat = 0;
                self.m_sto_frac = 0.;
            } else {
                let m_ldro: LdroMode = if ldro_mode_tmp == LdroMode::AUTO {
                    if (1_usize << self.m_sf) as f32 * 1e3 / self.m_bw as f32 > LDRO_MAX_DURATION_MS
                    {
                        LdroMode::ENABLE
                    } else {
                        LdroMode::DISABLE
                    }
                } else {
                    ldro_mode_tmp
                };

                self.m_symb_numb = 8
                    + ((2 * m_pay_len - self.m_sf
                        + 2
                        + (!self.m_impl_head) as usize * 5
                        + if m_has_crc { 4 } else { 0 }) as f64
                        / (self.m_sf - 2 * m_ldro as usize) as f64)
                        .ceil() as usize
                        * (4 + m_cr);
                self.m_received_head = true;
                frame_info.insert(String::from("is_header"), Pmt::Bool(false));
                frame_info.insert(String::from("symb_numb"), Pmt::Usize(self.m_symb_numb));
                frame_info.remove("ldro_mode");
                frame_info.insert(String::from("ldro"), Pmt::Bool(m_ldro as usize != 0));
                frame_info.insert(String::from("timestamp"), Pmt::U64(self.origin_timestamp.elapsed().as_nanos() as u64));
                let frame_info_pmt = Pmt::MapStrPmt(frame_info);
                self.tag_from_msg_handler_to_work_channel
                    .0
                    .try_send(frame_info_pmt)
                    .unwrap();
            }
        } else {
            warn!("noise_est pmt was not a Map/Dict");
        }
        Ok(Pmt::Null)
    }

   
}

#[async_trait]
impl Kernel for PhaseDifference {
    async fn work(
        &mut self,
        _io: &mut WorkIo,
        sio: &mut StreamIo,
        _mio: &mut MessageIo<Self>,
        _b: &mut BlockMeta,
    ) -> Result<()> {
        let input0 = sio.input(0).slice::<Complex32>();
        let input1 = sio.input(1).slice::<Complex32>();
        let out = sio.output(0).slice::<Complex32>();

        

        let n = std::cmp::min(input0.len(), input1.len());
        let n = std::cmp::min(n, out.len());

        for i in 0..n {
            let phase0 = input0[i].arg();
            let phase1 = input1[i].arg();
            out[i] = phase0 - phase1;
        }

        sio.input(0).consume(n);
        sio.input(1).consume(n);
        sio.output(0).produce(n);

        Ok(())
    }
}