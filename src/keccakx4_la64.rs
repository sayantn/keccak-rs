use core::arch::loongarch64::*;
use core::mem::transmute;

#[inline(always)]
unsafe fn xor(a: v4i64, b: v4i64) -> v4i64 {
    transmute(lasx_xvor_v(transmute(a), transmute(b)))
}

macro_rules! rotate_left {
    ($x: expr, $n: expr) => {
        lasx_xvrotri_d::<{ 64 - $n }>($x)
    };
}

#[inline(always)]
unsafe fn chi(a: v4i64, b: v4i64, c: v4i64) -> v4i64 {
    transmute(lasx_xvxor_v(transmute(a), lasx_xvandn_v(transmute(b), transmute(c))))
}

#[inline]
#[target_feature(enable = "lasx")]
pub unsafe fn interleave4([a, b, c, d]: [u64; 4]) -> v4i64 {
    transmute([a, b, c, d])
}

#[inline]
#[target_feature(enable = "lasx")]
pub unsafe fn uninterleave4(src: v4i64) -> [u64; 4] {
    transmute(src)
}

/// this function both interleaves and un-interleaves, and is faster than [interleave2]
#[inline]
#[target_feature(enable = "lasx")]
pub unsafe fn fast_interleave4([a, b, c, d]: [v4i64; 4]) -> [v4i64; 4] {
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

#[target_feature(enable = "lasx")]
pub unsafe fn keccak_p_parallel4<const ROUNDS: usize>(state: &[v4i64; 25]) -> [v4i64; 25] {
    crate::keccak_impl!(state, lasx_xvreplgr2vr_d);
}