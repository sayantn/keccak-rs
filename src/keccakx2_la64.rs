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
pub unsafe fn keccak_p_parallel2<const ROUNDS: usize>(state: &[v2i64; 25]) -> [v2i64; 25] {
    crate::keccak_impl!(state, lsx_vreplgr2vr_d);
}