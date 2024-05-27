// Implementation taken from XKCP

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::ops::BitXor;

use crate::ROUND_CONSTANTS;

const MOVE_THETA_PREV: __m512i = setr(4, 0, 1, 2, 3, 5, 6, 7);
const MOVE_THETA_NEXT: __m512i = setr(1, 2, 3, 4, 0, 5, 6, 7);

const RHO_B: __m512i = setr(0, 1, 62, 28, 27, 0, 0, 0);
const RHO_G: __m512i = setr(36, 44, 6, 55, 20, 0, 0, 0);
const RHO_K: __m512i = setr(3, 10, 43, 25, 39, 0, 0, 0);
const RHO_M: __m512i = setr(41, 45, 15, 21, 8, 0, 0, 0);
const RHO_S: __m512i = setr(18, 2, 61, 56, 14, 0, 0, 0);
const PI_1B: __m512i = setr(0, 3, 1, 4, 2, 5, 6, 7);
const PI_1G: __m512i = setr(1, 4, 2, 0, 3, 5, 6, 7);
const PI_1K: __m512i = setr(2, 0, 3, 1, 4, 5, 6, 7);
const PI_1M: __m512i = setr(3, 1, 4, 2, 0, 5, 6, 7);
const PI_1S: __m512i = setr(4, 2, 0, 3, 1, 5, 6, 7);
const PI_2S1: __m512i = setr(0, 1, 2, 3, 4, 5, 0 + 8, 2 + 8);
const PI_2S2: __m512i = setr(0, 1, 2, 3, 4, 5, 1 + 8, 3 + 8);
const PI_2BG: __m512i = setr(0, 1, 0 + 8, 1 + 8, 6, 5, 6, 7);
const PI_2KM: __m512i = setr(2, 3, 2 + 8, 3 + 8, 7, 5, 6, 7);
const PI_2S3: __m512i = setr(4, 5, 4 + 8, 5 + 8, 4, 5, 6, 7);

#[derive(Debug, Copy, Clone)]
pub struct KeccakState(__m512i, __m512i, __m512i, __m512i, __m512i);

impl PartialEq for KeccakState {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let acc = xor(self.0, other.0);
            let acc = _mm512_or_si512(acc, xor(self.1, other.1));
            let acc = _mm512_or_si512(acc, xor(self.2, other.2));
            let acc = _mm512_or_si512(acc, xor(self.3, other.3));
            let acc = _mm512_or_si512(acc, xor(self.4, other.4));
            _mm512_test_epi64_mask(acc, acc) == 0xff
        }
    }
}

impl Eq for KeccakState {}

impl From<[u64; 25]> for KeccakState {
    fn from(value: [u64; 25]) -> Self {
        unsafe {
            Self(
                load_plane(&value[0..]),
                load_plane(&value[5..]),
                load_plane(&value[10..]),
                load_plane(&value[15..]),
                load_plane(&value[20..]),
            )
        }
    }
}

impl From<KeccakState> for [u64; 25] {
    fn from(value: KeccakState) -> Self {
        let mut ret = [0; 25];
        unsafe {
            store_plane(&mut ret[0..], value.0);
            store_plane(&mut ret[5..], value.1);
            store_plane(&mut ret[10..], value.2);
            store_plane(&mut ret[15..], value.3);
            store_plane(&mut ret[20..], value.4);
        }
        ret
    }
}

impl<const LANES: usize> BitXor<[u64; LANES]> for KeccakState {
    type Output = Self;

    fn bitxor(self, rhs: [u64; LANES]) -> Self::Output {
        assert!(LANES > 0);
        assert!(LANES < 24);
        let mask = 0x1f1f1f1f1f_u64 & ((1_u64 << LANES) - 1);
        unsafe {
            Self(
                xor(self.0, _mm512_maskz_loadu_epi64((mask & 0xff) as __mmask8, rhs[0..].as_ptr().cast())),
                xor(self.1, _mm512_maskz_loadu_epi64(((mask >> 8) & 0xff) as __mmask8, rhs[5..].as_ptr().cast())),
                xor(self.2, _mm512_maskz_loadu_epi64(((mask >> 16) & 0xff) as __mmask8, rhs[10..].as_ptr().cast())),
                xor(self.3, _mm512_maskz_loadu_epi64(((mask >> 24) & 0xff) as __mmask8, rhs[15..].as_ptr().cast())),
                xor(self.4, _mm512_maskz_loadu_epi64(((mask >> 32) & 0xff) as __mmask8, rhs[25..].as_ptr().cast())),
            )
        }
    }
}

const fn set(a: u64) -> __m512i {
    setr(a as i64, 0, 0, 0, 0, 0, 0, 0)
}

const fn setr(a: i64, b: i64, c: i64, d: i64, e: i64, f: i64, g: i64, h: i64) -> __m512i {
    unsafe { core::mem::transmute([a, b, c, d, e, f, g, h]) }
}

#[inline(always)]
unsafe fn xor(a: __m512i, b: __m512i) -> __m512i {
    _mm512_xor_si512(a, b)
}

#[inline(always)]
unsafe fn xor3(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0x96>(a, b, c)
}

#[inline(always)]
unsafe fn chi(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0xd2>(a, b, c)
}

#[inline(always)]
unsafe fn load_plane(a: &[u64]) -> __m512i {
    _mm512_maskz_loadu_epi64(0x1f, a.as_ptr().cast())
}

#[inline(always)]
unsafe fn store_plane(a: &mut [u64], v: __m512i) {
    _mm512_mask_storeu_epi64(a.as_mut_ptr().cast(), 0x1f, v)
}

impl KeccakState {
    pub fn keccak_p<const ROUNDS: usize>(&self) -> Self {
        assert!(ROUNDS <= 24);
        unsafe {
            let Self(mut b, mut g, mut k, mut m, mut s) = self;

            for i in 24 - ROUNDS..24 {
                // Theta
                let b0 = xor3(xor3(b, g, k), m, s);
                let b1 = _mm512_permutexvar_epi64(MOVE_THETA_PREV, b0);
                let b0 = _mm512_permutexvar_epi64(MOVE_THETA_NEXT, b0);
                let b0 = _mm512_rol_epi64::<1>(b0);

                b = xor3(b, b0, b1);
                g = xor3(g, b0, b1);
                k = xor3(k, b0, b1);
                m = xor3(m, b0, b1);
                s = xor3(s, b0, b1);

                // Rho
                b = _mm512_rolv_epi64(b, RHO_B);
                g = _mm512_rolv_epi64(g, RHO_G);
                k = _mm512_rolv_epi64(k, RHO_K);
                m = _mm512_rolv_epi64(m, RHO_M);
                s = _mm512_rolv_epi64(s, RHO_S);

                // Pi - Part 1
                let b0 = _mm512_permutexvar_epi64(PI_1B, b);
                let b1 = _mm512_permutexvar_epi64(PI_1G, g);
                let b2 = _mm512_permutexvar_epi64(PI_1K, k);
                let b3 = _mm512_permutexvar_epi64(PI_1M, m);
                let b4 = _mm512_permutexvar_epi64(PI_1S, s);

                // Chi
                b = chi(b0, b1, b2);
                g = chi(b1, b2, b3);
                k = chi(b2, b3, b4);
                m = chi(b3, b4, b0);
                s = chi(b4, b0, b1);

                // Iota
                b = _mm512_xor_si512(b, set(ROUND_CONSTANTS[i]));

                // Pi - Part 2
                let b0 = _mm512_unpacklo_epi64(b, g);
                let b1 = _mm512_unpacklo_epi64(k, m);
                let b0 = _mm512_permutex2var_epi64(b0, PI_2S1, s);
                let b2 = _mm512_unpackhi_epi64(b, g);
                let b3 = _mm512_unpackhi_epi64(k, m);
                let b2 = _mm512_permutex2var_epi64(b2, PI_2S2, s);

                b = _mm512_permutex2var_epi64(b0, PI_2BG, b1);
                g = _mm512_permutex2var_epi64(b2, PI_2BG, b3);
                k = _mm512_permutex2var_epi64(b0, PI_2KM, b1);
                m = _mm512_permutex2var_epi64(b2, PI_2KM, b3);
                let b0 = _mm512_permutex2var_epi64(b0, PI_2S3, b1);
                s = _mm512_mask_blend_epi64(0x10, b0, s);
            }

            Self(b, g, k, m, s)
        }
    }
}
