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
pub unsafe fn interleave_state2(state: &[[u64; 25]; 2]) -> [uint64x2_t; 25] {
    let interleave = |offset| {
        fast_interleave2([
            vld1q_u64(state[0][offset..].as_ptr().cast()),
            vld1q_u64(state[1][offset..].as_ptr().cast()),
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

#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
pub unsafe fn uninterleave_state2(state: &[uint64x2_t; 25]) -> [[u64; 25]; 2] {
    let mut dst = [[0; 25]; 2];
    let mut store_uninterleaved = |offset| {
        let [a, b] = fast_interleave2([state[offset + 0], state[offset + 1]]);
        vst1q_u64(dst[0][offset..].as_mut_ptr().cast(), a);
        vst1q_u64(dst[1][offset..].as_mut_ptr().cast(), b);
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

#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
pub unsafe fn keccak_p_parallel2<const ROUNDS: usize>(
    state: &[uint64x2_t; 25],
) -> [uint64x2_t; 25] {
    crate::keccak_impl!(state, vdupq_n_u64);
}

#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
pub unsafe fn keccak_f_parallel2(state: &[uint64x2_t; 25]) -> [uint64x2_t; 25] {
    keccak_p_parallel2::<24>(state)
}
