// Implementation taken from XKCP

use crate::ROUND_CONSTANTS;
use core::arch::x86_64::*;
use core::mem;
use core::ops::BitXor;

#[rustfmt::skip]
const THETA_PERM: [__m512i; 4] = unsafe { mem::transmute([
    4_u64, 0, 1, 2, 3, 5, 6, 7,
    3, 4, 0, 1, 2, 5, 6, 7,
    2, 3, 4, 0, 1, 5, 6, 7,
    1, 2, 3, 4, 0, 5, 6, 7
]) };

#[rustfmt::skip]
const RHOTATES: [__m512i; 10] = unsafe { mem::transmute([
    0_u64, 1, 62, 28, 27, 0, 0, 0,
    36, 44, 6, 55, 20, 0, 0, 0,
    3, 10, 43, 25, 39, 0, 0, 0,
    41, 45, 15, 21, 8, 0, 0, 0,
    18, 2, 61, 56, 14, 0, 0, 0,
    0, 44, 43, 21, 14, 0, 0, 0,
    18, 1, 6, 25, 8, 0, 0, 0,
    41, 2, 62, 55, 39, 0, 0, 0,
    3, 45, 61, 28, 20, 0, 0, 0,
    36, 10, 15, 56, 27, 0, 0, 0
]) };

#[rustfmt::skip]
const PI_PERM: [__m512i; 5] = unsafe { mem::transmute([
    0_u64, 3, 1, 4, 2, 5, 6, 7,
    1, 4, 2, 0, 3, 5, 6, 7,
    2, 0, 3, 1, 4, 5, 6, 7,
    3, 1, 4, 2, 0, 5, 6, 7,
    4, 2, 0, 3, 1, 5, 6, 7
]) };

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
unsafe fn perm(a: __m512i, b: __m512i) -> __m512i {
    _mm512_permutexvar_epi64(a, b)
}

#[inline(always)]
unsafe fn build(a: __m512i, b: __m512i, c: __m512i, d: __m512i, e: __m512i) -> __m512i {
    let acc = _mm512_mask_blend_epi64(0x02, a, b);
    let acc = _mm512_mask_blend_epi64(0x04, acc, c);
    let acc = _mm512_mask_blend_epi64(0x08, acc, d);
    _mm512_mask_blend_epi64(0x10, acc, e)
}

#[derive(Debug, Copy, Clone)]
pub struct KeccakState(__m512i, __m512i, __m512i, __m512i, __m512i);

impl PartialEq for KeccakState {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let acc = xor(self.0, other.0);
            let acc = _mm512_ternarylogic_epi64::<0xf6>(acc, self.1, other.1);
            let acc = _mm512_ternarylogic_epi64::<0xf6>(acc, self.2, other.2);
            let acc = _mm512_ternarylogic_epi64::<0xf6>(acc, self.3, other.3);
            let acc = _mm512_ternarylogic_epi64::<0xf6>(acc, self.4, other.4);
            _mm512_test_epi64_mask(acc, acc) & 0x1f == 0x1f
        }
    }
}

impl Eq for KeccakState {}

impl From<[u64; 25]> for KeccakState {
    fn from(value: [u64; 25]) -> Self {
        unsafe {
            Self(
                _mm512_maskz_loadu_epi64(0x1f, value[0..].as_ptr().cast()),
                _mm512_maskz_loadu_epi64(0x1f, value[5..].as_ptr().cast()),
                _mm512_maskz_loadu_epi64(0x1f, value[10..].as_ptr().cast()),
                _mm512_maskz_loadu_epi64(0x1f, value[15..].as_ptr().cast()),
                _mm512_maskz_loadu_epi64(0x1f, value[20..].as_ptr().cast()),
            )
        }
    }
}

impl From<KeccakState> for [u64; 25] {
    fn from(value: KeccakState) -> Self {
        let mut ret = [0; 25];
        unsafe {
            _mm512_mask_storeu_epi64(ret[0..].as_mut_ptr().cast(), 0x1f, value.0);
            _mm512_mask_storeu_epi64(ret[5..].as_mut_ptr().cast(), 0x1f, value.1);
            _mm512_mask_storeu_epi64(ret[10..].as_mut_ptr().cast(), 0x1f, value.2);
            _mm512_mask_storeu_epi64(ret[15..].as_mut_ptr().cast(), 0x1f, value.3);
            _mm512_mask_storeu_epi64(ret[20..].as_mut_ptr().cast(), 0x1f, value.4);
        }
        ret
    }
}

impl<const LANES: usize> BitXor<[u64; LANES]> for KeccakState {
    type Output = KeccakState;

    fn bitxor(self, rhs: [u64; LANES]) -> Self::Output {
        assert!(LANES > 0 && LANES <= 24);
        let ptr = rhs.as_ptr().cast();
        unsafe {
            if LANES <= 5 {
                let mask = 0x1f >> (5 - LANES);
                Self(xor(self.0, _mm512_maskz_loadu_epi64(mask, ptr)), self.1, self.2, self.3, self.4)
            } else if LANES <= 10 {
                let mask = 0x1f >> (10 - LANES);
                Self(
                    xor(self.0, _mm512_maskz_loadu_epi64(0x1f, ptr)),
                    xor(self.1, _mm512_maskz_loadu_epi64(mask, ptr.add(5))),
                    self.2,
                    self.3,
                    self.4,
                )
            } else if LANES <= 15 {
                let mask = 0x1f >> (15 - LANES);
                Self(
                    xor(self.0, _mm512_maskz_loadu_epi64(0x1f, ptr)),
                    xor(self.1, _mm512_maskz_loadu_epi64(0x1f, ptr.add(5))),
                    xor(self.2, _mm512_maskz_loadu_epi64(mask, ptr.add(10))),
                    self.3,
                    self.4,
                )
            } else if LANES <= 20 {
                let mask = 0x1f >> (20 - LANES);
                Self(
                    xor(self.0, _mm512_maskz_loadu_epi64(0x1f, ptr)),
                    xor(self.1, _mm512_maskz_loadu_epi64(0x1f, ptr.add(5))),
                    xor(self.2, _mm512_maskz_loadu_epi64(0x1f, ptr.add(10))),
                    xor(self.3, _mm512_maskz_loadu_epi64(mask, ptr.add(15))),
                    self.4,
                )
            } else {
                let mask = 0x1f >> (25 - LANES);
                Self(
                    xor(self.0, _mm512_maskz_loadu_epi64(0x1f, ptr)),
                    xor(self.1, _mm512_maskz_loadu_epi64(0x1f, ptr.add(5))),
                    xor(self.2, _mm512_maskz_loadu_epi64(0x1f, ptr.add(10))),
                    xor(self.3, _mm512_maskz_loadu_epi64(0x1f, ptr.add(15))),
                    xor(self.4, _mm512_maskz_loadu_epi64(mask, ptr.add(20))),
                )
            }
        }
    }
}

impl KeccakState {
    pub fn keccak_p<const ROUNDS: usize>(&self) -> Self {
        let Self(mut s0, mut s1, mut s2, mut s3, mut s4) = self;

        unsafe {
            let mut r = 24 - ROUNDS;
            while r < 24 {
                // Theta + Rho + Pi - Even Round

                let t5 = xor3(xor3(s0, s1, s2), s3, s4);
                let t6 = perm(THETA_PERM[3], _mm512_rol_epi64::<1>(t5));
                let t5 = perm(THETA_PERM[0], t5);

                s0 = perm(PI_PERM[0], _mm512_rolv_epi64(xor3(t5, t6, s0), RHOTATES[0]));
                s1 = perm(PI_PERM[1], _mm512_rolv_epi64(xor3(t5, t6, s1), RHOTATES[1]));
                s2 = perm(PI_PERM[2], _mm512_rolv_epi64(xor3(t5, t6, s2), RHOTATES[2]));
                s3 = perm(PI_PERM[3], _mm512_rolv_epi64(xor3(t5, t6, s3), RHOTATES[3]));
                s4 = perm(PI_PERM[4], _mm512_rolv_epi64(xor3(t5, t6, s4), RHOTATES[4]));

                // Chi

                let a0 = chi(s0, s1, s2);
                let a1 = chi(s1, s2, s3);
                let a2 = chi(s2, s3, s4);
                let a3 = chi(s3, s4, s0);
                let a4 = chi(s4, s0, s1);

                // Iota

                let a0 = _mm512_mask_xor_epi64(a0, 0x01, a0, _mm512_loadu_epi64(ROUND_CONSTANTS[r + 0..].as_ptr().cast()));

                if r == 23 {
                    // Harmonize Last Round

                    let a1 = perm(THETA_PERM[0], a1);
                    let a2 = perm(THETA_PERM[1], a2);
                    let a3 = perm(THETA_PERM[2], a3);
                    let a4 = perm(THETA_PERM[3], a4);

                    s0 = build(a0, a1, a2, a3, a4);
                    s1 = perm(THETA_PERM[3], build(a1, a2, a3, a4, a0));
                    s2 = perm(THETA_PERM[2], build(a2, a3, a4, a0, a1));
                    s3 = perm(THETA_PERM[1], build(a3, a4, a0, a1, a2));
                    s4 = perm(THETA_PERM[0], build(a4, a0, a1, a2, a3));

                    break;
                }

                // Harmonize Rounds

                s0 = build(a0, a1, a2, a3, a4);
                s1 = perm(THETA_PERM[0], build(a1, a2, a3, a4, a0));
                s2 = perm(THETA_PERM[1], build(a2, a3, a4, a0, a1));
                s3 = perm(THETA_PERM[2], build(a3, a4, a0, a1, a2));
                s4 = perm(THETA_PERM[3], build(a4, a0, a1, a2, a3));

                // Theta + Rho - Odd Round

                let t5 = xor3(xor3(s0, s1, s2), s3, s4);
                let t6 = perm(THETA_PERM[3], _mm512_rol_epi64::<1>(t5));
                let t5 = perm(THETA_PERM[0], t5);

                let a0 = _mm512_rolv_epi64(xor3(t5, t6, s0), RHOTATES[5]);
                let a1 = _mm512_rolv_epi64(xor3(t5, t6, s3), RHOTATES[8]);
                let a2 = _mm512_rolv_epi64(xor3(t5, t6, s1), RHOTATES[6]);
                let a3 = _mm512_rolv_epi64(xor3(t5, t6, s4), RHOTATES[9]);
                let a4 = _mm512_rolv_epi64(xor3(t5, t6, s2), RHOTATES[7]);

                // Pi + Chi

                s0 = chi(a0, perm(THETA_PERM[3], a0), perm(THETA_PERM[0], a0));
                s1 = chi(perm(THETA_PERM[1], a1), perm(THETA_PERM[0], a1), a1);
                s2 = chi(perm(THETA_PERM[3], a2), perm(THETA_PERM[2], a2), perm(THETA_PERM[1], a2));
                s3 = chi(perm(THETA_PERM[0], a3), a3, perm(THETA_PERM[3], a3));
                s4 = chi(perm(THETA_PERM[2], a4), perm(THETA_PERM[1], a4), perm(THETA_PERM[0], a4));

                // Iota

                s0 = _mm512_mask_xor_epi64(s0, 0x01, s0, _mm512_loadu_epi64(ROUND_CONSTANTS[r + 1..].as_ptr().cast()));

                r += 2;
            }
        }

        Self(s0, s1, s2, s3, s4)
    }
}

// k6 = 0xffff
// k1 = k6 >> 15 = 0x01
// k6 = k6 >> 11 = 0x1f
// k2 = k1 << 1 = 0x02
// k3 = k1 << 2 = 0x04
// k4 = k1 << 3 = 0x08
// k5 = k1 << 4 = 0x10

// 13..=16 = THETA_PERM[0..4]
// 22..=31 = RHOTATES_0[0..10];
// 17..=21 = PI_PERM[0..5]
