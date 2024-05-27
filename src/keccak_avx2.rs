use crate::ROUND_CONSTANTS;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::mem;
use core::ops::BitXor;

#[rustfmt::skip]
const LEFT_SHIFTS: [__m256i; 6] = unsafe { mem::transmute([
    1_u64, 62, 28, 27,
    3, 18, 36, 41,
    45, 6, 56, 39,
    10, 61, 55, 8,
    2, 15, 25, 20,
    44, 43, 21, 14
]) };

#[rustfmt::skip]
const RIGHT_SHIFTS:[__m256i; 6] = unsafe { mem::transmute([
    63_u64, 2, 36, 37,
    61, 46, 28, 23,
    19, 58, 8, 25,
    54, 3, 9, 56,
    62, 49, 39, 44,
    20, 21, 43, 50
]) };

#[rustfmt::skip]
const MAPS:[__m128i; 5] = unsafe { mem::transmute([
    10, 20, 5, 15,
    16, 7, 23, 14,
    11, 22, 8, 19,
    21, 17, 13, 9,
    6, 12, 18, 24
]) };

#[derive(Debug, Copy, Clone)]
pub struct KeccakState(__m256i, __m256i, __m256i, __m256i, __m256i, __m256i, __m256i);

impl PartialEq for KeccakState {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let acc = xor(self.0, other.0);
            let acc = _mm256_or_si256(acc, xor(self.1, other.1));
            let acc = _mm256_or_si256(acc, xor(self.2, other.2));
            let acc = _mm256_or_si256(acc, xor(self.3, other.3));
            let acc = _mm256_or_si256(acc, xor(self.4, other.4));
            let acc = _mm256_or_si256(acc, xor(self.5, other.5));
            let acc = _mm256_or_si256(acc, xor(self.6, other.6));
            _mm256_testz_si256(acc, acc) == 1
        }
    }
}

impl Eq for KeccakState {}

impl From<[u64; 25]> for KeccakState {
    fn from(value: [u64; 25]) -> Self {
        unsafe {
            let all = _mm256_set1_epi64x(-1);
            Self(
                _mm256_set1_epi64x(value[0] as i64),
                _mm256_loadu_si256(value[1..].as_ptr().cast()),
                gather(&value, MAPS[0], all),
                gather(&value, MAPS[1], all),
                gather(&value, MAPS[2], all),
                gather(&value, MAPS[3], all),
                gather(&value, MAPS[4], all),
            )
        }
    }
}

impl From<KeccakState> for [u64; 25] {
    fn from(value: KeccakState) -> Self {
        unsafe {
            let s00 = _mm256_extract_epi64::<0>(value.0) as u64;
            let [s01, s02, s03, s04] = mem::transmute(value.1);
            let [s10, s20, s05, s15] = mem::transmute(value.2);
            let [s16, s07, s23, s14] = mem::transmute(value.3);
            let [s11, s22, s08, s19] = mem::transmute(value.4);
            let [s21, s17, s13, s09] = mem::transmute(value.5);
            let [s06, s12, s18, s24] = mem::transmute(value.6);

            #[rustfmt::skip]
            return [
                s00, s01, s02, s03, s04,
                s05, s06, s07, s08, s09,
                s10, s11, s12, s13, s14,
                s15, s16, s17, s18, s19,
                s20, s21, s22, s23, s24,
            ];
        }
    }
}

macro_rules! mask {
    ($lanes:ident: $($elem:expr),*) => {
        [$( if $lanes > $elem { -1_i64 } else { 0 } ),*]
    };
}

impl<const LANES: usize> BitXor<[u64; LANES]> for KeccakState {
    type Output = KeccakState;

    fn bitxor(self, rhs: [u64; LANES]) -> Self::Output {
        assert!(LANES > 0);
        assert!(LANES < 24);
        unsafe {
            if LANES < 5 {
                #[rustfmt::skip]
                const MASKS: [__m256i; 4] = unsafe { mem::transmute([
                    0_i64, 0, 0, 0,
                    -1, 0, 0, 0,
                    -1, -1, 0, 0,
                    -1, -1, -1, 0
                ]) };
                Self(
                    xor(self.0, _mm256_set1_epi64x(rhs[0] as i64)),
                    xor(self.1, _mm256_maskload_epi64(rhs[1..].as_ptr().cast(), MASKS[LANES - 1])),
                    self.2,
                    self.3,
                    self.4,
                    self.5,
                    self.6,
                )
            } else {
                #[rustfmt::skip]
                let masks: [__m256i; 5] = mem::transmute(mask!(LANES:
                    10, 20, 5, 15,
                    16, 7, 23, 14,
                    11, 22, 8, 19,
                    21, 17, 13, 9,
                    6, 12, 18, 24
                ));
                Self(
                    xor(self.0, _mm256_set1_epi64x(rhs[0] as i64)),
                    xor(self.1, _mm256_loadu_si256(rhs[1..].as_ptr().cast())),
                    xor(self.2, gather(&rhs, MAPS[0], masks[0])),
                    xor(self.3, gather(&rhs, MAPS[1], masks[1])),
                    xor(self.4, gather(&rhs, MAPS[2], masks[2])),
                    xor(self.5, gather(&rhs, MAPS[3], masks[3])),
                    xor(self.6, gather(&rhs, MAPS[4], masks[4])),
                )
            }
        }
    }
}

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
}

#[inline(always)]
unsafe fn rolv(a: __m256i, idx: usize) -> __m256i {
    _mm256_or_si256(_mm256_sllv_epi64(a, LEFT_SHIFTS[idx]), _mm256_srlv_epi64(a, RIGHT_SHIFTS[idx]))
}

#[inline(always)]
unsafe fn rol1(a: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_slli_epi64::<1>(a), _mm256_srli_epi64::<63>(a))
}

#[inline(always)]
unsafe fn perm<const PERM: i32>(a: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<PERM>(a)
}

#[inline(always)]
unsafe fn blend<const BLEND: i32>(a: __m256i, b: __m256i) -> __m256i {
    _mm256_blend_epi32::<BLEND>(a, b)
}

#[inline(always)]
unsafe fn and_not(a: __m256i, b: __m256i) -> __m256i {
    _mm256_andnot_si256(a, b)
}

/// takes 0:63 from a, 64:127 from b, 128:191 from c, 192:255 from d
#[inline(always)]
unsafe fn build(a: __m256i, b: __m256i, c: __m256i, d: __m256i) -> __m256i {
    blend::<0xc0>(blend::<0x30>(blend::<0x0c>(a, b), c), d)
}

#[inline(always)]
unsafe fn gather(slice: &[u64], map: __m128i, mask: __m256i) -> __m256i {
    _mm256_mask_i32gather_epi64::<8>(_mm256_setzero_si256(), slice.as_ptr().cast(), map, mask)
}

impl KeccakState {
    pub fn keccak_p<const ROUNDS: usize>(&self) -> KeccakState {
        unsafe {
            let Self(mut t0, mut t1, mut t2, mut t3, mut t4, mut t5, mut t6) = self;

            for i in 24 - ROUNDS..24 {
                let t12 = xor(xor(xor(t1, t3), t4), xor(t5, t6));
                let t11 = perm::<0x93>(t12);
                let t13 = xor(t2, _mm256_shuffle_epi32::<0x4e>(t2));

                let t8 = rol1(t12);

                let t15 = perm::<0x39>(t8);
                let t14 = perm::<0>(xor(t11, t8));

                let t13 = xor(xor(t13, t0), perm::<0x4e>(t13));
                let t8 = rol1(t13);

                let t15 = xor(blend::<0x03>(t11, t13), blend::<0xc0>(t15, t8));

                t0 = xor(t14, t0);
                t1 = xor(t15, t1);
                let t9 = rolv(t1, 0);
                t2 = rolv(xor(t14, t2), 1);
                t3 = rolv(xor(t15, t3), 2);
                t4 = rolv(xor(t15, t4), 3);
                t5 = rolv(xor(t15, t5), 4);
                t6 = xor(t15, t6);
                let t8 = rolv(t6, 5);

                let t10 = perm::<0x8d>(t2);
                let t11 = perm::<0x8d>(t3);
                let t12 = perm::<0x1b>(t4);
                let t13 = perm::<0x72>(t5);

                let t7 = and_not(t8, _mm256_bsrli_epi128::<8>(t8));
                t1 = and_not(
                    blend::<0xc0>(perm::<0x39>(t8), t0),
                    blend::<0x30>(perm::<0x1e>(t8), t0),
                );
                t2 = and_not(build(t11, t12, t13, t10), build(t13, t11, t10, t12));
                t3 = and_not(build(t9, t13, t11, t12), build(t11, t9, t12, t13));
                t4 = and_not(build(t13, t10, t12, t9), build(t12, t13, t9, t10));
                t5 = and_not(build(t10, t11, t9, t13), build(t9, t10, t13, t11));
                t6 = and_not(build(t12, t9, t10, t11), build(t10, t12, t11, t9));

                t0 = xor(perm::<0>(t7), t0);
                t1 = xor(t8, t1);
                t2 = xor(t9, t2);
                t3 = perm::<0x1b>(xor(t10, t3));
                t4 = xor(t11, t4);
                t5 = perm::<0x8d>(xor(t12, t5));
                t6 = perm::<0x72>(xor(t13, t6));

                // Iota

                t0 = xor(t0, _mm256_set1_epi64x(ROUND_CONSTANTS[i] as i64));
            }

            Self(t0, t1, t2, t3, t4, t5, t6)
        }
    }
}
