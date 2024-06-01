use core::arch::loongarch64::*;
use core::mem::transmute;

#[inline(always)]
unsafe fn xor(a: v2i64, b: v2i64) -> v2i64 {
    transmute(lsx_vxor_v(transmute(a), transmute(b)))
}

macro_rules! rotate_left {
    ($x: expr, $n: expr) => {
        lsx_vrotri_d::<{ 64 - $n }>($x)
    };
}

#[inline(always)]
unsafe fn chi(a: v2i64, b: v2i64, c: v2i64) -> v2i64 {
    transmute(lsx_vxor_v(transmute(a), lsx_vandn_v(transmute(b), transmute(c))))
}

#[inline]
#[target_feature(enable = "lsx")]
pub unsafe fn interleave2([a, b]: [u64; 2]) -> v2i64 {
    transmute([a, b])
}

#[inline]
#[target_feature(enable = "lsx")]
pub unsafe fn uninterleave2(src: v2i64) -> [u64; 2] {
    transmute(src)
}

/// this function both interleaves and un-interleaves, and is faster than [interleave2]
#[inline]
#[target_feature(enable = "lsx")]
pub unsafe fn fast_interleave2([a, b]: [v2i64; 2]) -> [v2i64; 2] {
    [lsx_vilvl_d(b, a), lsx_vilvh_d(b, a)]
}

#[target_feature(enable = "lsx")]
pub unsafe fn interleave_state2(state: &[[u64; 25]; 2]) -> [v2i64; 25] {
    let interleave = |offset| {
        fast_interleave2([
            transmute(lsx_vld::<0>(state[0][offset..].as_ptr().cast())),
            transmute(lsx_vld::<0>(state[1][offset..].as_ptr().cast())),
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

#[target_feature(enable = "lsx")]
pub unsafe fn uninterleave_state2(state: &[v2i64; 25]) -> [[u64; 25]; 2] {
    let mut dst = [[0; 25]; 2];
    let mut store_uninterleaved = |offset| {
        let [a, b] = fast_interleave2([state[offset + 0], state[offset + 1]]);
        lsx_vst::<0>(transmute(a), dst[0][offset..].as_mut_ptr().cast());
        lsx_vst::<0>(transmute(b), dst[1][offset..].as_mut_ptr().cast());
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

#[target_feature(enable = "lsx")]
pub unsafe fn keccak_p_parallel2<const ROUNDS: usize>(state: &[v2i64; 25]) -> [v2i64; 25] {
    crate::keccak_impl!(state, lsx_vreplgr2vr_d);
}

#[target_feature(enable = "lsx")]
pub unsafe fn keccak_f_parallel2(state: &[v2i64; 25]) -> [v2i64; 25] {
    keccak_p_parallel2::<24>(state)
}
