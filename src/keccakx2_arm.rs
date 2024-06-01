#[cfg(not(target_arch = "arm"))]
use core::arch::aarch64::*;
#[cfg(target_arch = "arm")]
use core::arch::arm::*;

#[inline(always)]
unsafe fn xor(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    veorq_u64(a, b)
}

macro_rules! rotate_left {
    ($x: expr, $n: expr) => {{
        let a = $x;
        vorrq_u64(vshlq_n_u64::<{ $n }>(a), vshrq_n_u64::<{ 64 - $n }>(a))
    }};
}

#[inline(always)]
unsafe fn chi(a: uint64x2_t, b: uint64x2_t, c: uint64x2_t) -> uint64x2_t {
    xor(a, vbicq_u64(c, b))
}

#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
pub unsafe fn interleave2([a, b]: [u64; 2]) -> uint64x2_t {
    vcombine_u64(vcreate_u64(a), vcreate_u64(b))
}

#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
pub unsafe fn uninterleave2(src: uint64x2_t) -> [u64; 2] {
    [vgetq_lane_u64::<0>(src), vgetq_lane_u64::<1>(src)]
}

/// this function both interleaves and un-interleaves, and is faster than [interleave2]
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
pub unsafe fn fast_interleave2([a, b]: [uint64x2_t; 2]) -> [uint64x2_t; 2] {
    #[cfg(target_arch = "arm")]
    return [
        vsetq_lane_u64::<1>(vgetq_lane_u64::<0>(b), a),
        vsetq_lane_u64::<0>(vgetq_lane_u64::<1>(a), b),
    ];
    #[cfg(not(target_arch = "arm"))]
    return [vtrn1q_u64(a, b), vtrn2q_u64(a, b)];
}


#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
pub unsafe fn keccak_p_parallel2<const ROUNDS: usize>(
    state: &[uint64x2_t; 25],
) -> [uint64x2_t; 25] {
    crate::keccak_impl!(state, vdupq_n_u64);
}