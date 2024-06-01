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
pub unsafe fn keccak_p_parallel4<const ROUNDS: usize>(state: &[__m256i; 25]) -> [__m256i; 25] {
    crate::keccak_impl!(state, _mm256_set1_epi64x);
}