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
pub fn interleave_state2(state: &[[u64; 25]; 2]) -> [v128; 25] {
    let interleave = |offset| unsafe {
        fast_interleave2([
            v128_load(state[0][offset..].as_ptr().cast()),
            v128_load(state[1][offset..].as_ptr().cast()),
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

#[target_feature(enable = "simd128")]
pub fn uninterleave_state2(state: &[v128; 25]) -> [[u64; 25]; 2] {
    let mut dst = [[0; 25]; 2];
    let mut store_uninterleaved = |offset| unsafe {
        let [a, b] = fast_interleave2([state[offset + 0], state[offset + 1]]);
        v128_store(dst[0][offset..].as_mut_ptr().cast(), a);
        v128_store(dst[1][offset..].as_mut_ptr().cast(), b);
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

#[target_feature(enable = "simd128")]
pub fn keccak_p_parallel2<const ROUNDS: usize>(state: &[v128; 25]) -> [v128; 25] {
    crate::keccak_impl!(state, u64x2_splat);
}

#[target_feature(enable = "simd128")]
pub fn keccak_f_parallel2(state: &[v128; 25]) -> [v128; 25] {
    keccak_p_parallel2::<24>(state)
}
