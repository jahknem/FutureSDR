//
//  Copyright 2004,2005,2009 Free Software Foundation, Inc.
//
//  This file is part of GNU Radio
//
//  SPDX-License-Identifier: GPL-3.0-or-later
//
//

// Routines for designing optimal FIR filters.
//
// For a great intro to how all this stuff works, see section 6.6 of
// "Digital Signal Processing: A Practical Approach", Emmanuael C. Ifeachor
// and Barrie W. Jervis, Adison-Wesley, 1993.  ISBN 0-201-54413-X.

use crate::pm_remez::pm_remez;
use ordered_float::OrderedFloat;

// from . import filter_python as filter

//  ----------------------------------------------------------------

// ""
// "
// Builds a low pass filter.
//
// Args:
//     gain: Filter gain in the passband (linear)
//     Fs: Sampling rate (sps)
//     freq1: End of pass band (in Hz)
//     freq2: Start of stop band (in Hz)
//     passband_ripple_db: Pass band ripple in dB (should be small, < 1)
//     stopband_atten_db: Stop band attenuation in dB (should be large, >= 60)
//     nextra_taps: Extra taps to use in the filter (default=2)
// "
// ""
pub fn low_pass(
    gain: f64,
    Fs: usize,
    freq1: f64,
    freq2: f64,
    passband_ripple_db: f64,
    stopband_atten_db: f64,
    nextra_taps: Option<usize>,
) -> Vec<f64> {
    let nextra_taps = nextra_taps.unwrap_or(2);
    let passband_dev = passband_ripple_to_dev(passband_ripple_db);
    let stopband_dev = stopband_atten_to_dev(stopband_atten_db);
    let (n, fo, ao, w) = remezord(
        &[freq1, freq2],
        &[gain, 0.0_f64],
        &[passband_dev, stopband_dev],
        Some(Fs),
    );
    // # The
    // remezord
    // typically
    // under - estimates
    // the
    // filter
    // order, so
    // add
    // 2
    // taps
    // by
    // default
    pm_remez(n + nextra_taps, &fo, &ao, &w, "bandpass", 16)
}

// def band_pass(gain, Fs, freq_sb1, freq_pb1, freq_pb2, freq_sb2,
//               passband_ripple_db, stopband_atten_db,
//               nextra_taps=2):
//     """
//     Builds a band pass filter.
//
//     Args:
//         gain: Filter gain in the passband (linear)
//         Fs: Sampling rate (sps)
//         freq_sb1: End of stop band (in Hz)
//         freq_pb1: Start of pass band (in Hz)
//         freq_pb2: End of pass band (in Hz)
//         freq_sb2: Start of stop band (in Hz)
//         passband_ripple_db: Pass band ripple in dB (should be small, < 1)
//         stopband_atten_db: Stop band attenuation in dB (should be large, >= 60)
//         nextra_taps: Extra taps to use in the filter (default=2)
//     """
//     passband_dev = passband_ripple_to_dev(passband_ripple_db)
//     stopband_dev = stopband_atten_to_dev(stopband_atten_db)
//     desired_ampls = (0, gain, 0)
//     desired_freqs = [freq_sb1, freq_pb1, freq_pb2, freq_sb2]
//     desired_ripple = [stopband_dev, passband_dev, stopband_dev]
//     (n, fo, ao, w) = remezord(desired_freqs, desired_ampls,
//                               desired_ripple, Fs)
//     # The remezord typically under-estimates the filter order, so add 2 taps by default
//     taps = filter.pm_remez(n + nextra_taps, fo, ao, w, "bandpass")
//     return taps
//
//
// def complex_band_pass(gain, Fs, freq_sb1, freq_pb1, freq_pb2, freq_sb2,
//                       passband_ripple_db, stopband_atten_db,
//                       nextra_taps=2):
//     """
//     Builds a band pass filter with complex taps by making an LPF and
//     spinning it up to the right center frequency
//
//     Args:
//         gain: Filter gain in the passband (linear)
//         Fs: Sampling rate (sps)
//         freq_sb1: End of stop band (in Hz)
//         freq_pb1: Start of pass band (in Hz)
//         freq_pb2: End of pass band (in Hz)
//         freq_sb2: Start of stop band (in Hz)
//         passband_ripple_db: Pass band ripple in dB (should be small, < 1)
//         stopband_atten_db: Stop band attenuation in dB (should be large, >= 60)
//         nextra_taps: Extra taps to use in the filter (default=2)
//     """
//     center_freq = (freq_pb2 + freq_pb1) / 2.0
//     lp_pb = (freq_pb2 - center_freq) / 1.0
//     lp_sb = freq_sb2 - center_freq
//     lptaps = low_pass(gain, Fs, lp_pb, lp_sb, passband_ripple_db,
//                       stopband_atten_db, nextra_taps)
//     spinner = [cmath.exp(2j * cmath.pi * center_freq / Fs * i)
//                for i in range(len(lptaps))]
//     taps = [s * t for s, t in zip(spinner, lptaps)]
//     return taps
//
//
// def complex_band_reject(gain, Fs, freq_pb1, freq_sb1, freq_sb2, freq_pb2,
//                         passband_ripple_db, stopband_atten_db,
//                         nextra_taps=2):
//     """
//     Builds a band reject filter with complex taps by making an HPF and
//     spinning it up to the right center frequency
//
//     Args:
//         gain: Filter gain in the passband (linear)
//         Fs: Sampling rate (sps)
//         freq_pb1: End of pass band (in Hz)
//         freq_sb1: Start of stop band (in Hz)
//         freq_sb2: End of stop band (in Hz)
//         freq_pb2: Start of pass band (in Hz)
//         passband_ripple_db: Pass band ripple in dB (should be small, < 1)
//         stopband_atten_db: Stop band attenuation in dB (should be large, >= 60)
//         nextra_taps: Extra taps to use in the filter (default=2)
//     """
//     center_freq = (freq_sb2 + freq_sb1) / 2.0
//     hp_pb = (freq_pb2 - center_freq) / 1.0
//     hp_sb = freq_sb2 - center_freq
//     hptaps = high_pass(gain, Fs, hp_sb, hp_pb, passband_ripple_db,
//                        stopband_atten_db, nextra_taps)
//     spinner = [cmath.exp(2j * cmath.pi * center_freq / Fs * i)
//                for i in range(len(hptaps))]
//     taps = [s * t for s, t in zip(spinner, hptaps)]
//     return taps
//
//
// def band_reject(gain, Fs, freq_pb1, freq_sb1, freq_sb2, freq_pb2,
//                 passband_ripple_db, stopband_atten_db,
//                 nextra_taps=2):
//     """
//     Builds a band reject filter
//     spinning it up to the right center frequency
//
//     Args:
//         gain: Filter gain in the passband (linear)
//         Fs: Sampling rate (sps)
//         freq_pb1: End of pass band (in Hz)
//         freq_sb1: Start of stop band (in Hz)
//         freq_sb2: End of stop band (in Hz)
//         freq_pb2: Start of pass band (in Hz)
//         passband_ripple_db: Pass band ripple in dB (should be small, < 1)
//         stopband_atten_db: Stop band attenuation in dB (should be large, >= 60)
//         nextra_taps: Extra taps to use in the filter (default=2)
//     """
//     passband_dev = passband_ripple_to_dev(passband_ripple_db)
//     stopband_dev = stopband_atten_to_dev(stopband_atten_db)
//     desired_ampls = (gain, 0, gain)
//     desired_freqs = [freq_pb1, freq_sb1, freq_sb2, freq_pb2]
//     desired_ripple = [passband_dev, stopband_dev, passband_dev]
//     (n, fo, ao, w) = remezord(desired_freqs, desired_ampls,
//                               desired_ripple, Fs)
//     # Make sure we use an odd number of taps
//     if((n + nextra_taps) % 2 == 1):
//         n += 1
//     # The remezord typically under-estimates the filter order, so add 2 taps by default
//     taps = filter.pm_remez(n + nextra_taps, fo, ao, w, "bandpass")
//     return taps
//
//
// def high_pass(gain, Fs, freq1, freq2, passband_ripple_db, stopband_atten_db,
//               nextra_taps=2):
//     """
//     Builds a high pass filter.
//
//     Args:
//         gain: Filter gain in the passband (linear)
//         Fs: Sampling rate (sps)
//         freq1: End of stop band (in Hz)
//         freq2: Start of pass band (in Hz)
//         passband_ripple_db: Pass band ripple in dB (should be small, < 1)
//         stopband_atten_db: Stop band attenuation in dB (should be large, >= 60)
//         nextra_taps: Extra taps to use in the filter (default=2)
//     """
//     passband_dev = passband_ripple_to_dev(passband_ripple_db)
//     stopband_dev = stopband_atten_to_dev(stopband_atten_db)
//     desired_ampls = (0, 1)
//     (n, fo, ao, w) = remezord([freq1, freq2], desired_ampls,
//                               [stopband_dev, passband_dev], Fs)
//     # For a HPF, we need to use an odd number of taps
//     # In filter.remez, ntaps = n+1, so n must be even
//     if((n + nextra_taps) % 2 == 1):
//         n += 1
//
//     # The remezord typically under-estimates the filter order, so add 2 taps by default
//     taps = filter.pm_remez(n + nextra_taps, fo, ao, w, "bandpass")
//     return taps
//
// //  ----------------------------------------------------------------
//
//
fn stopband_atten_to_dev(atten_db: f64) -> f64 {
    // ""
    // "Convert a stopband attenuation in dB to an absolute value"
    // ""
    return 10.0_f64.powf(-atten_db / 20.);
}

fn passband_ripple_to_dev(ripple_db: f64) -> f64 {
    // ""
    // "Convert passband ripple spec expressed in dB to an absolute value"
    // ""
    return (10.0_f64.powf(ripple_db / 20.) - 1.) / (10.0_f64.powf(ripple_db / 20.) + 1.);
}

//  ----------------------------------------------------------------

fn remezord(
    fcuts: &[f64],
    mags: &[f64],
    devs: &[f64],
    fsamp: Option<usize>,
) -> (usize, Vec<f64>, Vec<f64>, Vec<f64>) {
    // '''
    // FIR order estimator (lowpass, highpass, bandpass, mulitiband).
    //
    // (n, fo, ao, w) = remezord (f, a, dev)
    // (n, fo, ao, w) = remezord (f, a, dev, fs)
    //
    // (n, fo, ao, w) = remezord (f, a, dev) finds the approximate order,
    // normalized frequency band edges, frequency band amplitudes, and
    // weights that meet input specifications f, a, and dev, to use with
    // the remez command.
    //
    // * f is a sequence of frequency band edges (between 0 and Fs/2, where
    //   Fs is the sampling frequency), and a is a sequence specifying the
    //   desired amplitude on the bands defined by f. The length of f is
    //   twice the length of a, minus 2. The desired function is
    //   piecewise constant.
    //
    // * dev is a sequence the same size as a that specifies the maximum
    //   allowable deviation or ripples between the frequency response
    //   and the desired amplitude of the output filter, for each band.
    //
    // Use remez with the resulting order n, frequency sequence fo,
    // amplitude response sequence ao, and weights w to design the filter b
    // which approximately meets the specifications given by remezord
    // input parameters f, a, and dev:
    //
    // b = remez (n, fo, ao, w)
    //
    // (n, fo, ao, w) = remezord (f, a, dev, Fs) specifies a sampling frequency Fs.
    //
    // Fs defaults to 2 Hz, implying a Nyquist frequency of 1 Hz. You can
    // therefore specify band edges scaled to a particular applications
    // sampling frequency.
    //
    // In some cases remezord underestimates the order n. If the filter
    // does not meet the specifications, try a higher order such as n+1
    // or n+2.
    // '''
    // get local copies
    let fsamp = fsamp.unwrap_or(2);
    // fcuts = fcuts[:]
    // mags = mags[:]
    // devs = devs[:]

    let fcuts: Vec<f64> = fcuts.iter().map(|&x| x / fsamp as f64).collect();

    let nf = fcuts.len();
    let nm = mags.len();
    let nd = devs.len();
    let nbands = nm;

    assert!(nm == nd, "Length of mags and devs must be equal");

    assert!(
        nf == 2 * (nbands - 1),
        "Length of f must be 2 * len (mags) - 2"
    );

    let devs: Vec<f64> = devs
        .iter()
        .zip(mags.iter())
        .map(|(&d, &m)| if m == 0. { d } else { d / m })
        .collect(); // if not stopband, get relative deviation

    // separate the passband and stopband edges
    let f1: Vec<f64> = fcuts.iter().step_by(2).copied().collect();
    let f2: Vec<f64> = fcuts[1..].iter().step_by(2).copied().collect();

    let mut n = 0;
    let mut min_delta: f64 = 2.;
    for i in 0..f1.len() {
        if f2[i] - f1[i] < min_delta {
            n = i;
            min_delta = f2[i] - f1[i];
        }
    }
    let l = if nbands == 2 {
        // lowpass or highpass case (use formula)
        lporder(f1[n], f2[n], devs[0], devs[1])
    } else {
        // bandpass or multipass case
        // try different lowpasses and take the worst one that
        //  goes through the BP specs
        let mut l_tmp: f64 = 0.;
        for i in 1..(nbands - 1) {
            let l1 = lporder(f1[i - 1], f2[i - 1], devs[i], devs[i - 1]);
            let l2 = lporder(f1[i], f2[i], devs[i], devs[i + 1]);
            l_tmp = l_tmp.max(l1.max(l2));
        }
        l_tmp
    };

    let n = l.ceil() as usize - 1; // need order, not length for remez

    // cook up remez compatible result
    let mut ff: Vec<f64> = fcuts.iter().copied().map(|x| 2. * x).collect();
    ff.push(1.);
    ff.insert(0, 0.);

    let aa = mags
        .iter()
        .zip(mags.iter())
        .fold(vec![], |mut vec, (&a_1, &a_2)| {
            vec.push(a_1);
            vec.push(a_2);
            vec
        });

    let max_dev = devs
        .iter()
        .map(|&x| OrderedFloat(x))
        .max()
        .unwrap()
        .into_inner();
    let wts: Vec<f64> = devs.iter().map(|&x| max_dev / x).collect();

    return (n, ff, aa, wts);
}

//  ----------------------------------------------------------------

fn lporder(freq1: f64, freq2: f64, delta_p: f64, delta_s: f64) -> f64 {
    // '''
    // FIR lowpass filter length estimator.  freq1 and freq2 are
    // normalized to the sampling frequency.  delta_p is the passband
    // deviation (ripple), delta_s is the stopband deviation (ripple).
    //
    // Note, this works for high pass filters too (freq1 > freq2), but
    // doesn't work well if the transition is near f == 0 or f == fs/2
    //
    // From Herrmann et al (1973), Practical design rules for optimum
    // finite impulse response filters.  Bell System Technical J., 52, 769-99
    // '''
    let df = (freq2 - freq1).abs();
    let ddp = delta_p.log10();
    let dds = delta_s.log10();

    let a1 = 5.309e-3;
    let a2 = 7.114e-2;
    let a3 = -4.761e-1;
    let a4 = -2.66e-3;
    let a5 = -5.941e-1;
    let a6 = -4.278e-1;

    let b1 = 11.01217;
    let b2 = 0.5124401;

    let t1 = a1 * ddp * ddp;
    let t2 = a2 * ddp;
    let t3 = a4 * ddp * ddp;
    let t4 = a5 * ddp;

    let dinf = ((t1 + t2 + a3) * dds) + (t3 + t4 + a6);
    let ff = b1 + b2 * (ddp - dds);

    dinf / df - ff * df + 1.
}

//
// def bporder(freq1, freq2, delta_p, delta_s):
//     '''
//     FIR bandpass filter length estimator.  freq1 and freq2 are
//     normalized to the sampling frequency.  delta_p is the passband
//     deviation (ripple), delta_s is the stopband deviation (ripple).
//
//     From Mintzer and Liu (1979)
//     '''
//     df = abs(freq2 - freq1)
//     ddp = math.log10(delta_p)
//     dds = math.log10(delta_s)
//
//     a1 = 0.01201
//     a2 = 0.09664
//     a3 = -0.51325
//     a4 = 0.00203
//     a5 = -0.57054
//     a6 = -0.44314
//
//     t1 = a1 * ddp * ddp
//     t2 = a2 * ddp
//     t3 = a4 * ddp * ddp
//     t4 = a5 * ddp
//
//     cinf = dds * (t1 + t2 + a3) + t3 + t4 + a6
//     ginf = -14.6 * math.log10(delta_p / delta_s) - 16.9
//     n = cinf / df + ginf * df + 1
//     return n
