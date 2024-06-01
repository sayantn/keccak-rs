#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;
#[cfg(target_arch = "wasm64")]
use core::arch::wasm64::*;

#[inline(always)]
fn xor(a: v128, b: v128) -> v128 {
    v128_xor(a, b)
}

macro_rules! rotate_left {
    ($x: expr, $n: expr) => {{
        let a = $x;
        v128_or(u64x2_shl(a, $n), u64x2_shr(a, 64 - $n))
    }};
}

#[inline(always)]
fn chi(a: v128, b: v128, c: v128) -> v128 {
    xor(a, v128_andnot(c, b))
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn interleave2([a, b]: [u64; 2]) -> v128 {
    u64x2(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
pub fn uninterleave2(src: v128) -> [u64; 2] {
    [u64x2_extract_lane::<0>(src), u64x2_extract_lane::<1>(src)]
}

/// this function both interleaves and un-interleaves, and is faster than [interleave2]
#[inline]
#[target_feature(enable = "simd128")]
pub fn fast_interleave2([a, b]: [v128; 2]) -> [v128; 2] {
    [u64x2_shuffle::<0, 2>(a, b), u64x2_shuffle::<1, 3>(a, b)]
}

#[target_feature(enable = "simd128")]
pub fn keccak_p_parallel2<const ROUNDS: usize>(state: &[v128; 25]) -> [v128; 25] {
    crate::keccak_impl!(state, u64x2_splat);
}