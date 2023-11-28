use crate::interpolator_taps::*;
use futuredsp::fir::NonResamplingFirKernel;
use futuresdr::num_complex::Complex32;
use num_traits::Num;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::Mul;

fn build_filters() -> Vec<Vec<f32>> {
    let mut filters: Vec<Vec<f32>> = vec![];

    filters.reserve(NSTEPS + 1);
    for i in 0..(NSTEPS + 1) {
        let taps_tmp: Vec<f32> = TAPS[i].as_slice().iter().map(|&x| x as f32).collect();
        filters.push(taps_tmp);
    }
    return filters;
}

/// \brief Compute intermediate samples between signal samples x(k//Ts)
/// \ingroup filter_primitive
///
/// \details
/// This implements a Minimum Mean Squared Error interpolator with
/// 8 taps. It is suitable for signals where the bandwidth of
/// interest B = 1/(4//Ts) Where Ts is the time between samples.
///
/// Although mu, the fractional delay, is specified as a float, it
/// is actually quantized. 0.0 <= mu <= 1.0. That is, mu is
/// quantized in the interpolate method to 32nd's of a sample.
///
/// For more information, in the GNU Radio source code, see:
/// \li gr-filter/lib/gen_interpolator_taps/README
/// \li gr-filter/lib/gen_interpolator_taps/praxis.txt
pub struct MmseFirInterpolator<'a, T> {
    filters: Vec<Vec<f32>>,
    phantom: PhantomData<&'a T>,
}

impl<T> MmseFirInterpolator<'_, T>
where
    T: Copy + Num + Sum<T> + Mul<f32, Output = T> + 'static,
{
    pub fn new() -> Self {
        MmseFirInterpolator::<'static, T> {
            filters: build_filters(),
            phantom: PhantomData,
        }
    }

    /// \brief compute a single interpolated output value.
    ///
    /// \p input must have ntaps() valid entries and be 8-byte aligned.
    /// input[0] .. input[ntaps() - 1] are referenced to compute the output value.
    /// \throws std::invalid_argument if input is not 8-byte aligned.
    ///
    /// \p mu must be in the range [0, 1] and specifies the fractional delay.
    ///
    /// \returns the interpolated input value.
    pub fn interpolate(&self, input: &[T], mu: f32) -> T {
        let imu: usize = (mu * NSTEPS as f32).round() as usize;

        if (imu < 0) || (imu > NSTEPS) {
            panic!("mmse_fir_interpolator_cc: imu out of bounds.");
        }

        input[..NTAPS]
            .iter()
            .zip(self.filters[imu].iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }

    pub fn get_n_lookahead() -> usize {
        NSTEPS - 1 // number of future input samples required to compute output sample for current input sample
    }
}
