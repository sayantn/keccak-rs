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
pub unsafe fn interleave_state4(state: &[[u64; 25]; 4]) -> [v4i64; 25] {
    let interleave = |offset| {
        fast_interleave4([
            transmute(lasx_xvld::<0>(state[0][offset..].as_ptr().cast())),
            transmute(lasx_xvld::<0>(state[1][offset..].as_ptr().cast())),
            transmute(lasx_xvld::<0>(state[2][offset..].as_ptr().cast())),
            transmute(lasx_xvld::<0>(state[3][offset..].as_ptr().cast())),
        ])
    };
    let [s00, s01, s02, s03] = interleave(0);
    let [s04, s05, s06, s07] = interleave(4);
    let [s08, s09, s10, s11] = interleave(8);
    let [s12, s13, s14, s15] = interleave(12);
    let [s16, s17, s18, s19] = interleave(16);
    let [s20, s21, s22, s23] = interleave(20);
    let s24 = interleave4([state[0][24], state[1][24], state[2][24], state[3][24]]);

    #[rustfmt::skip]
    return [
        s00, s01, s02, s03, s04,
        s05, s06, s07, s08, s09,
        s10, s11, s12, s13, s14,
        s15, s16, s17, s18, s19,
        s20, s21, s22, s23, s24,
    ];
}

#[target_feature(enable = "lasx")]
pub unsafe fn uninterleave_state4(state: &[v4i64; 25]) -> [[u64; 25]; 4] {
    let mut dst = [[0; 25]; 4];
    let mut store_uninterleaved = |offset| {
        let [a, b, c, d] = fast_interleave4([
            state[offset + 0],
            state[offset + 1],
            state[offset + 2],
            state[offset + 3],
        ]);
        lasx_xvst::<0>(transmute(a), dst[0][offset..].as_mut_ptr().cast());
        lasx_xvst::<0>(transmute(b), dst[1][offset..].as_mut_ptr().cast());
        lasx_xvst::<0>(transmute(c), dst[2][offset..].as_mut_ptr().cast());
        lasx_xvst::<0>(transmute(d), dst[3][offset..].as_mut_ptr().cast());
    };
    store_uninterleaved(0);
    store_uninterleaved(4);
    store_uninterleaved(8);
    store_uninterleaved(12);
    store_uninterleaved(16);
    store_uninterleaved(20);
    [dst[0][24], dst[1][24], dst[2][24], dst[3][24]] = uninterleave4(state[24]);

    dst
}

#[target_feature(enable = "lasx")]
pub unsafe fn keccak_p_parallel4<const ROUNDS: usize>(state: &[v4i64; 25]) -> [v4i64; 25] {
    crate::keccak_impl!(state, lasx_xvreplgr2vr_d);
}

#[target_feature(enable = "lasx")]
pub unsafe fn keccak_f_parallel4(state: &[v4i64; 25]) -> [v4i64; 25] {
    keccak_p_parallel4::<24>(state)
}
