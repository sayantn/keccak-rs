#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
}

macro_rules! rotate_left {
    ($x: expr, $n: expr) => {{
        let a = $x;
        _mm256_or_si256(_mm256_slli_epi64::<{ $n }>(a), _mm256_srli_epi64::<{ 64 - $n }>(a))
    }};
}

#[inline(always)]
unsafe fn chi(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    xor(a, _mm256_andnot_si256(b, c))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn interleave4([a, b, c, d]: [u64; 4]) -> __m256i {
    _mm256_setr_epi64x(a as i64, b as i64, c as i64, d as i64)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn uninterleave4(src: __m256i) -> [u64; 4] {
    [
        _mm256_extract_epi64::<0>(src) as u64,
        _mm256_extract_epi64::<1>(src) as u64,
        _mm256_extract_epi64::<2>(src) as u64,
        _mm256_extract_epi64::<3>(src) as u64,
    ]
}

/// this function both interleaves and un-interleaves, and is faster than [interleave2]
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn fast_interleave4([a, b, c, d]: [__m256i; 4]) -> [__m256i; 4] {
    let p = _mm256_permute2x128_si256::<0x20>(a, c);
    let q = _mm256_permute2x128_si256::<0x20>(b, d);
    let r = _mm256_permute2x128_si256::<0x31>(a, c);
    let s = _mm256_permute2x128_si256::<0x31>(b, d);

    [
        _mm256_unpacklo_epi64(p, q),
        _mm256_unpackhi_epi64(p, q),
        _mm256_unpacklo_epi64(r, s),
        _mm256_unpackhi_epi64(r, s),
    ]
}

#[target_feature(enable = "avx2")]
pub unsafe fn interleave_state4(state: &[[u64; 25]; 4]) -> [__m256i; 25] {
    let interleave = |offset| {
        fast_interleave4([
            _mm256_loadu_si256(state[0][offset..].as_ptr().cast()),
            _mm256_loadu_si256(state[1][offset..].as_ptr().cast()),
            _mm256_loadu_si256(state[2][offset..].as_ptr().cast()),
            _mm256_loadu_si256(state[3][offset..].as_ptr().cast()),
        ])
    };
    let [s00, s01, s02, s03] = interleave(0);
    let [s04, s05, s06, s07] = interleave(4);
    let [s08, s09, s10, s11] = interleave(8);
    let [s12, s13, s14, s15] = interleave(12);
    let [s16, s17, s18, s19] = interleave(16);
    let [s20, s21, s22, s23] = interleave(20);
    let s24 = interleave4([state[0][24], state[1][24], state[2][24], state[3][24]]);

    #[rustfmt::skip]
    return [
        s00, s01, s02, s03, s04,
        s05, s06, s07, s08, s09,
        s10, s11, s12, s13, s14,
        s15, s16, s17, s18, s19,
        s20, s21, s22, s23, s24,
    ];
}

#[target_feature(enable = "avx2")]
pub unsafe fn uninterleave_state4(state: &[__m256i; 25]) -> [[u64; 25]; 4] {
    let mut dst = [[0; 25]; 4];
    let mut store_uninterleaved = |offset| {
        let [a, b, c, d] = fast_interleave4([
            state[offset + 0],
            state[offset + 1],
            state[offset + 2],
            state[offset + 3],
        ]);
        _mm256_storeu_si256(dst[0][offset..].as_mut_ptr().cast(), a);
        _mm256_storeu_si256(dst[1][offset..].as_mut_ptr().cast(), b);
        _mm256_storeu_si256(dst[2][offset..].as_mut_ptr().cast(), c);
        _mm256_storeu_si256(dst[3][offset..].as_mut_ptr().cast(), d);
    };
    store_uninterleaved(0);
    store_uninterleaved(4);
    store_uninterleaved(8);
    store_uninterleaved(12);
    store_uninterleaved(16);
    store_uninterleaved(20);
    [dst[0][24], dst[1][24], dst[2][24], dst[3][24]] = uninterleave4(state[24]);

    dst
}

#[target_feature(enable = "avx2")]
pub unsafe fn keccak_p_parallel4<const ROUNDS: usize>(state: &[__m256i; 25]) -> [__m256i; 25] {
    crate::keccak_impl!(state, _mm256_set1_epi64x);
}

#[target_feature(enable = "avx2")]
pub unsafe fn keccak_f_parallel4(state: &[__m256i; 25]) -> [__m256i; 25] {
    keccak_p_parallel4::<24>(state)
}
