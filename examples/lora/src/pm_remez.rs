/**************************************************************************
 * Parks-McClellan algorithm for FIR filter design (C version)
 *-------------------------------------------------
 *  Copyright (c) 1995,1998  Jake Janovetz (janovetz@uiuc.edu)
 *  Copyright (c) 2004  Free Software Foundation, Inc.
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 *
 *  Sep 1999 - Paul Kienzle (pkienzle@cs.indiana.edu)
 *      Modified for use in octave as a replacement for the matlab function
 *      remez.mex.  In particular, magnitude responses are required for all
 *      band edges rather than one per band, griddensity is a parameter,
 *      and errors are returned rather than printed directly.
 *  Mar 2000 - Kai Habel (kahacjde@linux.zrz.tu-berlin.de)
 *      Change: ColumnVector x=arg(i).vector_value();
 *      to: ColumnVector x(arg(i).vector_value());
 *  There appear to be some problems with the routine search. See comments
 *  therein [search for PAK:].  I haven't looked closely at the rest
 *  of the code---it may also have some problems.
 *************************************************************************/

/*
 * This code was extracted from octave.sf.net, and wrapped with
 * GNU Radio glue.
 */

/*
 * This code was then ported to rust for use with FutureSDR.
 */

use futuresdr::log::warn;
use ordered_float::OrderedFloat;
use rustfft::num_traits::ToPrimitive;
use std::f64::consts::PI;

// #include <gnuradio/filter/pm_remez.h>
// #include <gnuradio/logger.h>
// #include <cassert>
// #include <cmath>
//
// #ifndef LOCAL_BUFFER
// #include <vector>
// #define LOCAL_BUFFER(T, buf, size)     \
//     std::vector<T> buf##_vector(size); \
//     T* buf = &(buf##_vector[0])
// #endif
//
// namespace gr {
// namespace filter {
//
// #define CONST const
const BANDPASS: usize = 1;
const DIFFERENTIATOR: usize = 2;
const HILBERT: usize = 3;
//
const NEGATIVE: bool = false;
const POSITIVE: bool = true;
//
// #define Pi 3.14159265358979323846
// #define Pi2 (2 * Pi)
//
const GRIDDENSITY: usize = 16;
const MAXITERATIONS: usize = 40;

/// create_dense_grid
///=================
///
/// Creates the dense grid of frequencies from the specified bands.
/// Also creates the Desired Frequency Response function (D[]) and
/// the Weight function (W[]) on that dense grid
///
///
/// INPUT:
/// ------
/// int      r        - 1/2 the number of filter coefficients
/// int      numtaps  - Number of taps in the resulting filter
/// int      numband  - Number of bands in user specification
/// double   bands[]  - User-specified band edges [2*numband]
/// double   des[]    - Desired response per band [2*numband]
/// double   weight[] - Weight per band [numband]
/// int      symmetry - Symmetry of filter - used for grid check
/// int      griddensity
///
/// OUTPUT:
/// -------
/// int    gridsize   - Number of elements in the dense frequency grid
/// double Grid[]     - Frequencies (0 to 0.5) on the dense grid [gridsize]
/// double D[]        - Desired response on the dense grid [gridsize]
/// double W[]        - Weight function on the dense grid [gridsize]
fn create_dense_grid(
    r: usize,
    numtaps: usize,
    numband: usize,
    bands: &[f64],
    des: &[f64],
    weight: &[f64],
    symmetry: bool,
    griddensity: usize,
    gridsize: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let delf = 0.5 / (griddensity as f64 * r as f64);

    /*
     * For differentiator, hilbert,
     *   symmetry is odd and Grid[0] = max(delf, bands[0])
     */
    let grid0 = if (symmetry == NEGATIVE) && (delf > bands[0]) {
        delf
    } else {
        bands[0]
    };

    let mut Grid = vec![0.; gridsize];
    let mut D = vec![0.; gridsize];
    let mut W = vec![0.; gridsize];
    let mut j: usize = 0;
    for band in 0..numband {
        let mut lowf = if band == 0 { grid0 } else { bands[2 * band] };
        let highf = bands[2 * band + 1];
        let k = ((highf - lowf) / delf + 0.5); /* .5 for rounding */
        for i in 0..(k as usize) {
            D[j] = des[2 * band] + i as f64 * (des[2 * band + 1] - des[2 * band]) / (k - 1.);
            W[j] = weight[band];
            Grid[j] = lowf;
            lowf += delf;
            j += 1;
        }
        Grid[j - 1] = highf;
    }

    /*
     * Similar to above, if odd symmetry, last grid point can't be .5
     *  - but, if there are even taps, leave the last grid point at .5
     */
    if (symmetry == NEGATIVE) && (Grid[gridsize - 1] > (0.5 - delf)) && (numtaps % 2) != 0 {
        Grid[gridsize - 1] = 0.5 - delf;
    }
    return (Grid, D, W);
}

/********************
 * initial_guess
 *==============
 * Places Extremal Frequencies evenly throughout the dense grid.
 *
 *
 * INPUT:
 * ------
 * int r        - 1/2 the number of filter coefficients
 * int gridsize - Number of elements in the dense frequency grid
 *
 * OUTPUT:
 * -------
 * int ext[]    - Extremal indexes to dense frequency grid [r+1]
 ********************/
fn initial_guess(r: usize, gridsize: usize) -> Vec<usize> {
    (0..r).map(|i| i * (gridsize - 1) / r).collect()
}

/***********************
 * calc_parms
 *===========
 *
 *
 * INPUT:
 * ------
 * int    r      - 1/2 the number of filter coefficients
 * int    Ext[]  - Extremal indexes to dense frequency grid [r+1]
 * double Grid[] - Frequencies (0 to 0.5) on the dense grid [gridsize]
 * double D[]    - Desired response on the dense grid [gridsize]
 * double W[]    - Weight function on the dense grid [gridsize]
 *
 * OUTPUT:
 * -------
 * double ad[]   - 'b' in Oppenheim & Schafer [r+1]
 * double x[]    - [r+1]
 * double y[]    - 'C' in Oppenheim & Schafer [r+1]
 ***********************/

fn calc_parms(
    r: usize,
    Ext: &[usize],
    Grid: &[f64],
    D: &[f64],
    W: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // int i, j, k, ld;
    // double sign, xi, delta, denom, numer;

    /*
     * Find x[]
     */
    let x: Vec<f64> = Ext.iter().map(|i| (2. * PI * Grid[*i]).cos()).collect();

    /*
     * Calculate ad[]  - Oppenheim & Schafer eq 7.132
     */
    let ld = (r - 1) / 15 + 1; /* Skips around to avoid round errors */
    let ad: Vec<f64> = (0..r)
        .map(|i| {
            let mut denom: f64 = 1.;
            let xi = x[i];
            for j in 0..ld {
                for k in (j..r).step_by(ld) {
                    if k != i {
                        denom *= 2.0 * (xi - x[k]);
                    }
                }
            }
            if denom.abs() < 0.00001 {
                denom = 0.00001;
            }
            1.0 / denom
        })
        .collect();

    /*
     * Calculate delta  - Oppenheim & Schafer eq 7.131
     */
    let mut numer: f64 = 0.;
    let mut denom: f64 = 0.;
    let mut sign: f64 = 1.;
    for i in 0..r {
        numer += ad[i] * D[Ext[i]];
        denom += sign * ad[i] / W[Ext[i]];
        sign = -sign;
    }
    let delta = numer / denom;
    sign = 1.;

    /*
     * Calculate y[]  - Oppenheim & Schafer eq 7.133b
     */
    let mut y = vec![0.; r];
    for i in 0..r {
        y[i] = D[Ext[i]] - sign * delta / W[Ext[i]];
        sign = -sign;
    }

    (ad, x, y)
}

/*********************
 * compute_A
 *==========
 * Using values calculated in calc_parms, compute_A calculates the
 * actual filter response at a given frequency (freq).  Uses
 * eq 7.133a from Oppenheim & Schafer.
 *
 *
 * INPUT:
 * ------
 * double freq - Frequency (0 to 0.5) at which to calculate A
 * int    r    - 1/2 the number of filter coefficients
 * double ad[] - 'b' in Oppenheim & Schafer [r+1]
 * double x[]  - [r+1]
 * double y[]  - 'C' in Oppenheim & Schafer [r+1]
 *
 * OUTPUT:
 * -------
 * Returns double value of A[freq]
 *********************/

fn compute_A(freq: f64, r: usize, ad: &[f64], x: &[f64], y: &[f64]) -> f64 {
    // int i;
    // double xc, c, denom, numer;

    let mut denom: f64 = 0.;
    let mut numer: f64 = 0.;
    let xc = (2. * PI * freq).cos();
    for i in 0..r {
        let mut c = xc - x[i];
        if c.abs() < 1.0e-7 {
            numer = y[i];
            denom = 1.;
            break;
        }
        c = ad[i] / c;
        denom += c;
        numer += c * y[i];
    }
    numer / denom
}

/************************
 * calc_error
 *===========
 * Calculates the Error function from the desired frequency response
 * on the dense grid (D[]), the weight function on the dense grid (W[]),
 * and the present response calculation (A[])
 *
 *
 * INPUT:
 * ------
 * int    r      - 1/2 the number of filter coefficients
 * double ad[]   - [r+1]
 * double x[]    - [r+1]
 * double y[]    - [r+1]
 * int gridsize  - Number of elements in the dense frequency grid
 * double Grid[] - Frequencies on the dense grid [gridsize]
 * double D[]    - Desired response on the dense grid [gridsize]
 * double W[]    - Weight function on the dense grid [gridsize]
 *
 * OUTPUT:
 * -------
 * double E[]    - Error function on dense grid [gridsize]
 ************************/

fn calc_error(
    r: usize,
    ad: &[f64],
    x: &[f64],
    y: &[f64],
    gridsize: usize,
    Grid: &[f64],
    D: &[f64],
    W: &[f64],
) -> Vec<f64> {
    (0..gridsize)
        .map(|i| W[i] * (D[i] - compute_A(Grid[i], r, ad, x, y)))
        .collect()
}

/************************
 * search
 *========
 * Searches for the maxima/minima of the error curve.  If more than
 * r+1 extrema are found, it uses the following heuristic (thanks
 * Chris Hanson):
 * 1) Adjacent non-alternating extrema deleted first.
 * 2) If there are more than one excess extrema, delete the
 *    one with the smallest error.  This will create a non-alternation
 *    condition that is fixed by 1).
 * 3) If there is exactly one excess extremum, delete the smaller
 *    of the first/last extremum
 *
 *
 * INPUT:
 * ------
 * int    r        - 1/2 the number of filter coefficients
 * int    Ext[]    - Indexes to Grid[] of extremal frequencies [r+1]
 * int    gridsize - Number of elements in the dense frequency grid
 * double E[]      - Array of error values.  [gridsize]
 * OUTPUT:
 * -------
 * int    Ext[]    - New indexes to extremal frequencies [r+1]
 ************************/
fn search(r: usize, Ext: &mut [usize], gridsize: usize, E: &[f64]) -> i8 {
    // int i, j, k, l, extra; /* Counters */
    // int up, alt;
    // int* foundExt; /* Array of found extremals */
    /*
     * Allocate enough space for found extremals.
     */
    // foundExt = (int*)malloc((2 * r) * sizeof(int));
    let mut k = 0;
    let mut foundExt = vec![0_usize; 2 * r];

    /*
     * Check for extremum at 0.
     */
    if ((E[0] > 0.0) && (E[0] > E[1])) || ((E[0] < 0.0) && (E[0] < E[1])) {
        foundExt[k] = 0;
        k += 1;
    }

    /*
     * Check for extrema inside dense grid
     */
    for i in 1..(gridsize - 1) {
        if ((E[i] >= E[i - 1]) && (E[i] > E[i + 1]) && (E[i] > 0.0))
            || ((E[i] <= E[i - 1]) && (E[i] < E[i + 1]) && (E[i] < 0.0))
        {
            // PAK: we sometimes get too many extremal frequencies
            if k >= 2 * r {
                return -3;
            }
            foundExt[k] = i;
            k += 1;
        }
    }

    /*
     * Check for extremum at 0.5
     */
    let j = gridsize - 1;
    if ((E[j] > 0.0) && (E[j] > E[j - 1])) || ((E[j] < 0.0) && (E[j] < E[j - 1])) {
        if k >= 2 * r {
            return -3;
        }
        foundExt[k] = j;
        k += 1;
    }

    // PAK: we sometimes get not enough extremal frequencies
    if k < r + 1 {
        return -2;
    }

    /*
     * Remove extra extremals
     */
    let mut extra = k - (r + 1);
    assert!(extra >= 0);

    while extra > 0 {
        let mut up = E[foundExt[0]] > 0.0; // first one is a maxima  vs  first one is a minima

        let mut l: usize = 0;
        let mut alt: bool = true;
        for j in 1..k {
            if E[foundExt[j]].abs() < E[foundExt[l]].abs() {
                l = j; /* new smallest error. */
            }
            if up && (E[foundExt[j]] < 0.0) {
                up = false; /* switch to a minima */
            } else if !up && (E[foundExt[j]] > 0.0) {
                up = true; /* switch to a maxima */
            } else {
                alt = false;
                // PAK: break now and you will delete the smallest overall
                // extremal.  If you want to delete the smallest of the
                // pair of non-alternating extremals, then you must do:
                //
                // if(fabs(E[foundExt[j]]) < fabs(E[foundExt[j-1]])) l=j;
                // else l=j-1;
                break; /* Ooops, found two non-alternating */
            } /* extrema.  Delete smallest of them */
        } /* if the loop finishes, all extrema are alternating */

        /*
         * If there's only one extremal and all are alternating,
         * delete the smallest of the first/last extremals.
         */
        if alt && (extra == 1) {
            if E[foundExt[k - 1]].abs() < E[foundExt[0]].abs() {
                /* Delete last extremal */
                l = k - 1;
            }
            // PAK: changed from l = foundExt[k-1];
            else {
                /* Delete first extremal */
                l = 0;
            }
            // PAK: changed from l = foundExt[0];
        }

        for j in l..(k - 1) {
            /* Loop that does the deletion */
            foundExt[j] = foundExt[j + 1];
            assert!(foundExt[j] < gridsize);
        }
        k -= 1;
        extra -= 1;
    }

    for i in 0..r {
        assert!(foundExt[i] < gridsize);
        Ext[i] = foundExt[i]; /* Copy found extremals to Ext[] */
    }
    return 0;
}

/*********************
 * freq_sample
 *============
 * Simple frequency sampling algorithm to determine the impulse
 * response h[] from A's found in compute_A
 *
 *
 * INPUT:
 * ------
 * int      N        - Number of filter coefficients
 * double   A[]      - Sample points of desired response [N/2]
 * int      symmetry - Symmetry of desired filter
 *
 * OUTPUT:
 * -------
 * double h[] - Impulse Response of final filter [N]
 *********************/
fn freq_sample(N: usize, A: &[f64], symm: bool) -> Vec<f64> {
    let M = (N - 1) as f64 / 2.0;
    if symm == POSITIVE {
        if (N % 2) != 0 {
            (0..N)
                .map(|n| {
                    let mut val = A[0];
                    let x = 2. * PI * (n as f64 - M) / N as f64;
                    for k in 1..(M as usize) {
                        val += 2.0 * A[k] * (x * k as f64).cos();
                    }
                    val / N as f64
                })
                .collect()
        } else {
            (0..N)
                .map(|n| {
                    let mut val = A[0];
                    let x = 2. * PI * (n as f64 - M) / N as f64;
                    for k in 1..(N / 2 - 1) {
                        val += 2.0 * A[k] * (x * k as f64).cos();
                    }
                    val / N as f64
                })
                .collect()
        }
    } else {
        if (N % 2) != 0 {
            (0..N)
                .map(|n| {
                    let mut val = 0.;
                    let x = 2. * PI * (n as f64 - M) / N as f64;
                    for k in 1..(M as usize) {
                        val += 2.0 * A[k] * (x * k as f64).sin();
                    }
                    val / N as f64
                })
                .collect()
        } else {
            (0..N)
                .map(|n| {
                    let mut val = A[N / 2] * (PI * (n as f64 - M)).sin();
                    let x = 2. * PI * (n as f64 - M) / N as f64;
                    for k in 1..(N / 2 - 1) {
                        val += 2.0 * A[k] * (x * k as f64).sin();
                    }
                    val / N as f64
                })
                .collect()
        }
    }
}

/*******************
 * is_done
 *========
 * Checks to see if the error function is small enough to consider
 * the result to have converged.
 *
 * INPUT:
 * ------
 * int    r     - 1/2 the number of filter coefficients
 * int    Ext[] - Indexes to extremal frequencies [r+1]
 * double E[]   - Error function on the dense grid [gridsize]
 *
 * OUTPUT:
 * -------
 * Returns 1 if the result converged
 * Returns 0 if the result has not converged
 ********************/

fn is_done(r: usize, Ext: &[usize], E: &[f64]) -> bool {
    let min = Ext
        .iter()
        .map(|i| E[*i].abs())
        .map(|x| OrderedFloat::<f64>(x))
        .min()
        .unwrap()
        .into_inner();
    let max = Ext
        .iter()
        .map(|i| E[*i].abs())
        .map(|x| OrderedFloat::<f64>(x))
        .max()
        .unwrap()
        .into_inner();
    return ((max - min) / max) < 0.0001;
}

/// remez
///=======
/// Calculates the optimal (in the Chebyshev/minimax sense)
/// FIR filter impulse response given a set of band edges,
/// the desired response on those bands, and the weight given to
/// the error in those bands.
///
/// INPUT:
/// ------
/// int     numtaps     - Number of filter coefficients
/// int     numband     - Number of bands in filter specification
/// double  bands[]     - User-specified band edges [2 * numband]
/// double  des[]       - User-specified band responses [2 * numband]
/// double  weight[]    - User-specified error weights [numband]
/// int     type        - Type of filter
///
/// OUTPUT:
/// -------
/// double h[]      - Impulse response of final filter [numtaps]
/// returns         - true on success, false on failure to converge
fn remez(
    numtaps: usize,
    numband: usize,
    bands: &[f64],
    des: &[f64],
    weight: &[f64],
    filter_type: usize,
    griddensity: usize,
) -> (Vec<f64>, i8) {
    // double *Grid, *W, *D, *E;
    // int i, iter, *Ext;
    // double *taps, c;
    // double *x, *y, *ad;
    // int symmetry;

    let symmetry = if filter_type == BANDPASS {
        POSITIVE
    } else {
        NEGATIVE
    };
    let mut r = numtaps / 2; /* number of extrema */
    if (numtaps % 2) != 0 && (symmetry == POSITIVE) {
        r += 1;
    }

    /*
     * Predict dense grid size in advance for memory allocation
     *   .5 is so we round up, not truncate
     */
    let mut gridsize: usize = 0;
    for i in 0..numband {
        gridsize +=
            2 * r * griddensity * ((bands[2 * i + 1] - bands[2 * i]) + 0.5).floor() as usize;
    }
    if symmetry == NEGATIVE {
        gridsize -= 1;
    }

    /*
     * Dynamically allocate memory for arrays with proper sizes
     */
    // Grid = (double*)malloc(gridsize * sizeof(double));
    // D = (double*)malloc(gridsize * sizeof(double));
    // W = (double*)malloc(gridsize * sizeof(double));
    // E = (double*)malloc(gridsize * sizeof(double));
    // Ext = (int*)malloc((r + 1) * sizeof(int));
    // taps = (double*)malloc((r + 1) * sizeof(double));
    // x = (double*)malloc((r + 1) * sizeof(double));
    // y = (double*)malloc((r + 1) * sizeof(double));
    // ad = (double*)malloc((r + 1) * sizeof(double));

    /*
     * Create dense frequency grid
     */
    let (Grid, mut D, mut W) = create_dense_grid(
        r,
        numtaps,
        numband,
        bands,
        des,
        weight,
        symmetry,
        griddensity,
        gridsize,
    );
    let mut Ext = initial_guess(r, gridsize);

    /*
     * For Differentiator: (fix grid)
     */
    if filter_type == DIFFERENTIATOR {
        for i in 0..gridsize {
            /* D[i] = D[i]*Grid[i]; */
            if D[i] > 0.0001 {
                W[i] = W[i] / Grid[i];
            }
        }
    }

    /*
     * For odd or Negative symmetry filters, alter the
     * D[] and W[] according to Parks McClellan
     */
    if symmetry == POSITIVE {
        if numtaps % 2 == 0 {
            for i in 0..gridsize {
                let c = (PI * Grid[i]).cos();
                D[i] /= c;
                W[i] *= c;
            }
        }
    } else {
        if numtaps % 2 != 0 {
            for i in 0..gridsize {
                let c = (2. * PI * Grid[i]).sin();
                D[i] /= c;
                W[i] *= c;
            }
        } else {
            for i in 0..gridsize {
                let c = (PI * Grid[i]).sin();
                D[i] /= c;
                W[i] *= c;
            }
        }
    }

    /*
     * Perform the Remez Exchange algorithm
     */
    let mut num_iter: usize = 0;
    for iter in 0..MAXITERATIONS {
        let (ad, x, y) = calc_parms(r, &Ext, &Grid, &D, &W);
        let E = calc_error(r, &ad, &x, &y, gridsize, &Grid, &D, &W);
        let err = search(r, &mut Ext, gridsize, &E);
        if err > 0 {
            return (vec![0.], err);
        }
        for i in 0..r {
            assert!(Ext[i] < gridsize);
        }
        num_iter = iter;
        if is_done(r, &Ext, &E) {
            break;
        }
    }

    let (ad, x, y) = calc_parms(r, &Ext, &Grid, &D, &W);

    /*
     * Find the 'taps' of the filter for use with Frequency
     * Sampling.  If odd or Negative symmetry, fix the taps
     * according to Parks McClellan
     */
    let taps: Vec<f64> = (0..(numtaps / 2))
        .map(|i| {
            let c: f64 = if symmetry == POSITIVE {
                if numtaps % 2 != 0 {
                    1.
                } else {
                    (PI * i as f64 / numtaps as f64).cos()
                }
            } else {
                if numtaps % 2 != 0 {
                    (2. * PI * i as f64 / numtaps as f64).sin()
                } else {
                    (PI * i as f64 / numtaps as f64).sin()
                }
            };
            compute_A(i as f64 / numtaps as f64, r, &ad, &x, &y) * c
        })
        .collect();

    /*
     * Frequency sampling design with calculated taps
     */
    let h = freq_sample(numtaps, &taps, symmetry);

    return (h, if num_iter < MAXITERATIONS { 0 } else { -1 });
}

//////////////////////////////////////////////////////////////////////////////
//
//			    GNU Radio interface
//
//////////////////////////////////////////////////////////////////////////////

fn punt(msg: &str) {
    warn!("pm_remez: {}", msg);
    panic!("{}", msg);
}

/// \brief Parks-McClellan FIR filter design using Remez algorithm.
/// \ingroup filter_design
///
/// \details
/// Calculates the optimal (in the Chebyshev/minimax sense) FIR
/// filter inpulse response given a set of band edges, the desired
/// response on those bands, and the weight given to the error in
/// those bands.
///
/// \param order         filter order (number of taps in the returned filter - 1)
/// \param bands         frequency at the band edges [ b1 e1 b2 e2 b3 e3 ...]
/// \param ampl  	    desired amplitude at the band edges [ a(b1) a(e1) a(b2) a(e2)
///...] \param error_weight  weighting applied to each band (usually 1) \param filter_type
///one of "bandpass", "hilbert" or "differentiator"
/// \param grid_density  determines how accurately the filter will be constructed. \
///			    The minimum value is 16; higher values are slower to compute.
///
/// Frequency is in the range [0, 1], with 1 being the Nyquist
/// frequency (Fs/2)
///
/// \returns vector of computed taps
///
/// \throws std::runtime_error if args are invalid or calculation
/// fails to converge.
pub fn pm_remez(
    order: usize,
    arg_bands: &[f64],
    arg_response: &[f64],
    arg_weight: &[f64],
    filter_type: &str,
    grid_density: usize,
) -> Vec<f64> {
    // let loggers_set_up = false;
    //
    // gr::logger_ptr logger, debug_logger;
    // if (!loggers_set_up) {
    //     gr::configure_default_loggers(logger, debug_logger, "pm_remez");
    //     loggers_set_up = true;
    // }
    let numtaps = order + 1;
    if numtaps < 4 {
        punt("number of taps must be >= 3");
    }

    let numbands = arg_bands.len() / 2;
    if numbands < 1 || arg_bands.len() % 2 == 1 {
        punt("must have an even number of band edges");
    }

    for i in 1..arg_bands.len() {
        if arg_bands[i] < arg_bands[i - 1] {
            punt("band edges must be nondecreasing");
        }
    }

    if arg_bands[0] < 0. || arg_bands[arg_bands.len() - 1] > 1. {
        punt("band edges must be in the range [0,1]");
    }

    // Divide by 2 to fit with the implementation that uses a
    // sample rate of [0, 0.5] instead of [0, 1.0]
    let bands: Vec<f64> = arg_bands.iter().map(|x| x / 2.).collect();

    if arg_response.len() != arg_bands.len() {
        punt("must have one response magnitude for each band edge");
    }

    let response = arg_response;

    let weight = if !arg_weight.is_empty() {
        if arg_weight.len() != numbands {
            punt("need one weight for each band [=length(band)/2]");
        }
        arg_weight.to_vec()
    } else {
        vec![1.0_f64; numbands]
    };

    let itype: usize = if filter_type == "bandpass" {
        BANDPASS
    } else if filter_type == "differentiator" {
        DIFFERENTIATOR
    } else if filter_type == "hilbert" {
        HILBERT
    } else {
        punt(&format!("unknown ftype '{}'", filter_type));
        0
    };

    if grid_density < 16 {
        punt("grid_density is too low; must be >= 16");
    }

    // LOCAL_BUFFER(double, coeff, numtaps + 5); // FIXME why + 5?
    let (coeff, err) = remez(
        numtaps,
        numbands,
        &bands,
        response,
        &weight,
        itype,
        grid_density,
    );

    if err == -1 {
        punt("failed to converge");
    }

    if err == -2 {
        punt("insufficient extremals -- cannot continue");
    }

    if err == -3 {
        punt("too many extremals -- cannot continue");
    }

    coeff
}
