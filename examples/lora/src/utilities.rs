// #include <iomanip>
// #include <numeric>
// #include <gnuradio/expj.h>
// #include <volk/volk.h>
// #include <algorithm>
use futuresdr::num_complex::Complex32;
use ordered_float::OrderedFloat;
use rustfft::num_traits::Float;
use std::f32::consts::PI;
use std::ops::Mul;

// pub const RESET: &str = "\033[0m";
// pub const RED: &str = "\033[31m"; /* Red */
pub const MIN_SF: usize = 5; //minimum and maximum SF
pub const MAX_SF: usize = 12;

pub type LLR = f64; //< Log-Likelihood Ratio type
                    //typedef long double LLR; // 16 Bytes

#[repr(usize)]
#[derive(Debug, Copy, Clone)]
pub enum Symbol_type {
    VOID,
    UPCHIRP,
    SYNC_WORD,
    DOWNCHIRP,
    QUARTER_DOWN,
    PAYLOAD,
    UNDETERMINED,
}

pub const LDRO_MAX_DURATION_MS: f32 = 16.;
#[repr(usize)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ldro_mode {
    DISABLE,
    ENABLE,
    AUTO,
}
// /**
//  *  \brief  return the modulus a%b between 0 and (b-1)
//  */
// #[inline]
// inline long mod(long a, long b)
// { return (a%b+b)%b; }  // TODO ???

// inline double double_mod(double a, long b)
// { return fmod(fmod(a,b)+b,b);}

/**
 *  \brief  Convert an integer into a MSB first vector of bool
 *
 *  \param  integer
 *          The integer to convert
 *  \param  n_bits
 *          The output number of bits
 */
#[inline]
pub fn int2bool(integer: u32, n_bits: usize) -> Vec<bool> {
    let mut vec: Vec<bool> = vec![false; n_bits];
    let mut j = n_bits;
    for i in 0_usize..n_bits {
        j -= 1;
        vec[j] = ((integer >> i) & 1) != 0;
    }
    vec
}
/**
 *  \brief  Convert a MSB first vector of bool to a integer
 *
 *  \param  b
 *          The boolean vector to convert
 */
#[inline]
pub fn bool2int(b: Vec<bool>) -> u32 {
    b.iter()
        .map(|x| *x as u32)
        .zip((0_usize..b.len()).rev())
        .map(|(bit, order)| bit << order)
        .fold(0_u32, |acc, e| acc + e)
}

/**
 *  \brief  Return an modulated upchirp using s_f=bw
 *
 *  \param  chirp
 *          The pointer to the modulated upchirp
 *  \param  id
 *          The number used to modulate the chirp
 * \param   sf
 *          The spreading factor to use
 * \param os_factor
 *          The oversampling factor used to generate the upchirp
 */
#[inline]
pub fn build_upchirp(id: u32, sf: usize) -> Vec<Complex32> {
    let n = (1 << sf) as f32;
    let n_idx = 1 << sf;
    let n_fold = n - id as f32;
    let mut chirp = vec![Complex32::from(0.); 1 << sf];
    for i in 0..n_idx {
        let j = i as f32;
        if n < n_fold {
            chirp[n_idx] = Complex32::new(1.0, 0.0)
                * Complex32::from_polar(
                    1.,
                    2.0 * PI * (j * j / (2. * n) + (id as f32 / n as f32 - 0.5) * j),
                );
        } else {
            chirp[n_idx] = Complex32::new(1.0, 0.0)
                * Complex32::from_polar(
                    1.,
                    2.0 * PI * (j * j / (2. * n) + (id as f32 / n as f32 - 1.5) * j),
                );
        }
    }
    chirp
}

/**
 *  \brief  Return the reference chirps using s_f=bw
 *
 *  \param  upchirp
 *          The pointer to the reference upchirp
 *  \param  downchirp
 *          The pointer to the reference downchirp
 * \param   sf
 *          The spreading factor to use
 */
#[inline]
pub fn build_ref_chirps(sf: usize) -> (Vec<Complex32>, Vec<Complex32>) {
    let n: f64 = (1 << sf) as f64;
    let upchirp = build_upchirp(0, sf);
    let downchirp = volk_32fc_conjugate_32fc(&upchirp);
    (upchirp, downchirp)
}

//  // find most frequency number in vector
// inline int most_frequent(int arr[], int n)
// {
//     // Insert all elements in hash.
//     std::unordered_map<int, int> hash;
//     for (int i = 0; i < n; i++)
//         hash[arr[i]]++;
//
//     // find the max frequency
//     int max_count = 0, res = -1;
//     for (auto i : hash) {
//         if (max_count < i.second) {
//             res = i.first;
//             max_count = i.second;
//         }
//     }
//
//     return res;
// }

// inline std::string random_string(int Nbytes){
// const char* charmap = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
// const size_t charmapLength = strlen(charmap);
// auto generator = [&](){ return charmap[rand()%charmapLength]; };
// std::string result;
// result.reserve(Nbytes);
// std::generate_n(std::back_inserter(result), Nbytes, generator);
// return result;
//     }

pub fn argmax_float<T: Float>(input_slice: &[T]) -> usize {
    input_slice
        .iter()
        .map(|x| OrderedFloat::<T>(*x))
        .enumerate()
        .max_by(|(_, value0), (_, value1)| value0.cmp(value1))
        .map(|(idx, _)| idx)
        .unwrap_or(0_usize)
}

// TODO possibly limit size of input slices according to 'num_points' param from cpp code
pub fn volk_32fc_conjugate_32fc(a_vector: &Vec<Complex32>) -> Vec<Complex32> {
    // TODO use map and generic type
    let mut b_vector = vec![Complex32::from(0.); a_vector.len()];
    for i in 0_usize..a_vector.len() {
        let tmp: Complex32 = a_vector[i];
        b_vector[i] = tmp.conj();
    }
    b_vector
}

pub fn volk_32fc_x2_multiply_32fc<T: Copy + Mul<T, Output = T>>(
    input_slice_1: &[T],
    input_slice_2: &[T],
) -> Vec<T> {
    let tmp: Vec<T> = input_slice_1
        .iter()
        .zip(input_slice_2.iter())
        .map(|(x, y)| *x * *y)
        .collect();
    tmp
}

pub fn volk_32fc_magnitude_squared_32f(input_slice: &[Complex32]) -> Vec<f32> {
    let tmp: Vec<f32> = input_slice
        .iter()
        .map(|x| x.re * x.re + x.im * x.im)
        .collect();
    tmp
}
