phase_difference:

Save value on mio in into value 0/1. If other value is there, calculate. But only if total consumed is near.

m_sto_int

// HERE OUTPUT FOR THESE VARS: (in separate message output (PMT<hashmap>))
m_cfo_int:  // 
m_sto_frac: // 
k_hat:
(total_consumed_samples)

Total Time from boot to frame start = (1 / sampling freq) * (total_consumed_samples + sto_int + sto_frac)
sto_int = k_hat - m_cfo_int

toa delta = total time a - total time b (Betrag)

toa delta -> phasen differenz umrechnen

formula 2 from syloin paper for aoa



Manuell berechnen für statischen Winkel (indem ich Werte manuell statisch addiere (1 Antenne als 2 streams))



let up_sample = FirBuilder::new_resampling::<Complex32, Complex32>(STO_FRAC_DENOM.abs() as usize, 1);
let sampling_time_offset = Delay::<Complex32>::new((STO_INT + 23) * STO_FRAC_DENOM + STO_FRAC_NOM - 1); // -1 to compensate resampling delay of 1 sample (I guess...)
let down_sample = FirBuilder::new_resampling::<Complex32, Complex32>(1, STO_FRAC_DENOM.abs() as usize);

modulate [Circular::with_size(2 * 4 * 8192 * 4 * 8 * 16)]
up_sample > sampling_time_offset > down_sample > cfo_block >
frame_sync >



Duplicate using split block (add lambda function that maps from a to b and c)


Software modulation using tx.rs


m_sto_frac wo ein Header detected wird 
int m_cfo_int wo ein Header detected wird.
k_hat
STO_int (=k_hat - CFO_int)

frac wie viel innerhalb eines samples
int wie viel innerhalb eines frame


Sollte bei software only 0 sein.
