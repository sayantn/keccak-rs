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
pub unsafe fn keccak_p_parallel2<const ROUNDS: usize>(state: &[__m128i; 25]) -> [__m128i; 25] {
    crate::keccak_impl!(state, _mm_set1_epi64x);
}

#[target_feature(enable = "sse2")]
pub unsafe fn keccak_f_parallel2(state: &[__m128i; 25]) -> [__m128i; 25] {
    keccak_p_parallel2::<24>(state)
}
