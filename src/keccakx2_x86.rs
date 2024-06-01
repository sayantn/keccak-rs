#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
unsafe fn xor(a: __m128i, b: __m128i) -> __m128i {
    _mm_xor_si128(a, b)
}

macro_rules! rotate_left {
    ($x: expr, $n: expr) => {{
        let a = $x;
        _mm_or_si128(_mm_slli_epi64::<{ $n }>(a), _mm_srli_epi64::<{ 64 - $n }>(a))
    }};
}

#[inline(always)]
unsafe fn chi(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    xor(a, _mm_andnot_si128(b, c))
}

#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn interleave2([a, b]: [u64; 2]) -> __m128i {
    _mm_set_epi64x(b as i64, a as i64)
}

#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn uninterleave2(src: __m128i) -> [u64; 2] {
    let mut dst = [0; 2];
    _mm_storel_epi64(dst.as_mut_ptr().cast(), src);
    _mm_storeh_pd(dst[1..].as_mut_ptr().cast(), _mm_castsi128_pd(src));
    dst
}

/// this function both interleaves and un-interleaves, and is faster than [interleave2]
#[inline]
#[target_feature(enable = "sse2")]
pub unsafe fn fast_interleave2([a, b]: [__m128i; 2]) -> [__m128i; 2] {
    [_mm_unpacklo_epi64(a, b), _mm_unpackhi_epi64(a, b)]
}

#[target_feature(enable = "sse2")]
pub unsafe fn interleave_state2(state: &[[u64; 25]; 2]) -> [__m128i; 25] {
    let interleave = |offset| {
        fast_interleave2([
            _mm_loadu_si128(state[0][offset..].as_ptr().cast()),
            _mm_loadu_si128(state[1][offset..].as_ptr().cast()),
        ])
    };
    let [s00, s01] = interleave(0);
    let [s02, s03] = interleave(2);
    let [s04, s05] = interleave(4);
    let [s06, s07] = interleave(6);
    let [s08, s09] = interleave(8);
    let [s10, s11] = interleave(10);
    let [s12, s13] = interleave(12);
    let [s14, s15] = interleave(14);
    let [s16, s17] = interleave(16);
    let [s18, s19] = interleave(18);
    let [s20, s21] = interleave(20);
    let [s22, s23] = interleave(22);
    let s24 = interleave2([state[0][24], state[1][24]]);

    #[rustfmt::skip]
    return [
        s00, s01, s02, s03, s04,
        s05, s06, s07, s08, s09,
        s10, s11, s12, s13, s14,
        s15, s16, s17, s18, s19,
        s20, s21, s22, s23, s24,
    ];
}

#[target_feature(enable = "sse2")]
pub unsafe fn uninterleave_state2(state: &[__m128i; 25]) -> [[u64; 25]; 2] {
    let mut dst = [[0; 25]; 2];
    let mut store_uninterleaved = |offset| {
        let [a, b] = fast_interleave2([state[offset + 0], state[offset + 1]]);
        _mm_storeu_si128(dst[0][offset..].as_mut_ptr().cast(), a);
        _mm_storeu_si128(dst[1][offset..].as_mut_ptr().cast(), b);
    };
    store_uninterleaved(0);
    store_uninterleaved(2);
    store_uninterleaved(4);
    store_uninterleaved(6);
    store_uninterleaved(8);
    store_uninterleaved(10);
    store_uninterleaved(12);
    store_uninterleaved(14);
    store_uninterleaved(16);
    store_uninterleaved(18);
    store_uninterleaved(20);
    store_uninterleaved(22);
    [dst[0][24], dst[1][24]] = uninterleave2(state[24]);

    dst
}

#[target_feature(enable = "sse2")]
pub unsafe fn keccak_p_parallel2<const ROUNDS: usize>(state: &[__m128i; 25]) -> [__m128i; 25] {
    crate::keccak_impl!(state, _mm_set1_epi64x);
}

#[target_feature(enable = "sse2")]
pub unsafe fn keccak_f_parallel2(state: &[__m128i; 25]) -> [__m128i; 25] {
    keccak_p_parallel2::<24>(state)
}
