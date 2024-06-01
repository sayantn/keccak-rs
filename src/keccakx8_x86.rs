#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
unsafe fn xor(a: __m512i, b: __m512i) -> __m512i {
    _mm512_xor_si512(a, b)
}

macro_rules! rotate_left {
    ($x: expr, $n: expr) => {
        _mm512_rol_epi64::<{ $n }>($x)
    };
}

#[inline(always)]
unsafe fn chi(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    _mm512_ternarylogic_epi64::<0xd2>(a, b, c)
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn interleave8([a, b, c, d, e, f, g, h]: [u64; 8]) -> __m512i {
    _mm512_setr_epi64(
        a as i64, b as i64, c as i64, d as i64, e as i64, f as i64, g as i64, h as i64,
    )
}

#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn uninterleave8(src: __m512i) -> [u64; 8] {
    core::mem::transmute(src)
}

/// this function both interleaves and un-interleaves, and is faster than [interleave2]
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn fast_interleave8([a, b, c, d, e, f, g, h]: [__m512i; 8]) -> [__m512i; 8] {
    let p = _mm512_unpacklo_epi64(a, b);
    let q = _mm512_unpackhi_epi64(a, b);
    let r = _mm512_unpacklo_epi64(c, d);
    let s = _mm512_unpackhi_epi64(c, d);
    let t = _mm512_unpacklo_epi64(e, f);
    let u = _mm512_unpackhi_epi64(e, f);
    let v = _mm512_unpacklo_epi64(g, h);
    let w = _mm512_unpackhi_epi64(g, h);

    let a = _mm512_shuffle_i64x2::<0x88>(p, r);
    let b = _mm512_shuffle_i64x2::<0x88>(q, s);
    let c = _mm512_shuffle_i64x2::<0xdd>(p, r);
    let d = _mm512_shuffle_i64x2::<0xdd>(q, s);
    let e = _mm512_shuffle_i64x2::<0x88>(t, v);
    let f = _mm512_shuffle_i64x2::<0x88>(u, w);
    let g = _mm512_shuffle_i64x2::<0xdd>(t, v);
    let h = _mm512_shuffle_i64x2::<0xdd>(u, w);

    [
        _mm512_shuffle_i64x2::<0x88>(a, e),
        _mm512_shuffle_i64x2::<0x88>(b, f),
        _mm512_shuffle_i64x2::<0x88>(c, g),
        _mm512_shuffle_i64x2::<0x88>(d, h),
        _mm512_shuffle_i64x2::<0xdd>(a, e),
        _mm512_shuffle_i64x2::<0xdd>(b, f),
        _mm512_shuffle_i64x2::<0xdd>(c, g),
        _mm512_shuffle_i64x2::<0xdd>(d, h),
    ]
}

#[target_feature(enable = "avx512f")]
pub unsafe fn interleave_state8(state: &[[u64; 25]; 8]) -> [__m512i; 25] {
    let interleave = |offset| {
        fast_interleave8([
            _mm512_loadu_si512(state[0][offset..].as_ptr().cast()),
            _mm512_loadu_si512(state[1][offset..].as_ptr().cast()),
            _mm512_loadu_si512(state[2][offset..].as_ptr().cast()),
            _mm512_loadu_si512(state[3][offset..].as_ptr().cast()),
            _mm512_loadu_si512(state[4][offset..].as_ptr().cast()),
            _mm512_loadu_si512(state[5][offset..].as_ptr().cast()),
            _mm512_loadu_si512(state[6][offset..].as_ptr().cast()),
            _mm512_loadu_si512(state[7][offset..].as_ptr().cast()),
        ])
    };
    let [s00, s01, s02, s03, s04, s05, s06, s07] = interleave(0);
    let [s08, s09, s10, s11, s12, s13, s14, s15] = interleave(8);
    let [s16, s17, s18, s19, s20, s21, s22, s23] = interleave(16);
    let s24 = interleave8([
        state[0][24],
        state[1][24],
        state[2][24],
        state[3][24],
        state[4][24],
        state[5][24],
        state[6][24],
        state[7][24],
    ]);

    #[rustfmt::skip]
    return [
        s00, s01, s02, s03, s04,
        s05, s06, s07, s08, s09,
        s10, s11, s12, s13, s14,
        s15, s16, s17, s18, s19,
        s20, s21, s22, s23, s24,
    ];
}

#[target_feature(enable = "avx512f")]
pub unsafe fn uninterleave_state8(state: &[__m512i; 25]) -> [[u64; 25]; 8] {
    let mut dst = [[0; 25]; 8];
    let mut store_uninterleaved = |offset| {
        let [a, b, c, d, e, f, g, h] = fast_interleave8([
            state[offset + 0],
            state[offset + 1],
            state[offset + 2],
            state[offset + 3],
            state[offset + 4],
            state[offset + 5],
            state[offset + 6],
            state[offset + 7],
        ]);
        _mm512_storeu_si512(dst[0][offset..].as_mut_ptr().cast(), a);
        _mm512_storeu_si512(dst[1][offset..].as_mut_ptr().cast(), b);
        _mm512_storeu_si512(dst[2][offset..].as_mut_ptr().cast(), c);
        _mm512_storeu_si512(dst[3][offset..].as_mut_ptr().cast(), d);
        _mm512_storeu_si512(dst[4][offset..].as_mut_ptr().cast(), e);
        _mm512_storeu_si512(dst[5][offset..].as_mut_ptr().cast(), f);
        _mm512_storeu_si512(dst[6][offset..].as_mut_ptr().cast(), g);
        _mm512_storeu_si512(dst[7][offset..].as_mut_ptr().cast(), h);
    };
    store_uninterleaved(0);
    store_uninterleaved(8);
    store_uninterleaved(16);
    [
        dst[0][24], dst[1][24], dst[2][24], dst[3][24], dst[4][24], dst[5][24], dst[6][24],
        dst[7][24],
    ] = uninterleave8(state[24]);

    dst
}

#[target_feature(enable = "avx512f")]
pub unsafe fn keccak_p_parallel8<const ROUNDS: usize>(state: &[__m512i; 25]) -> [__m512i; 25] {
    crate::keccak_impl!(state, _mm512_set1_epi64);
}

#[target_feature(enable = "avx512f")]
pub unsafe fn keccak_f_parallel8(state: &[__m512i; 25]) -> [__m512i; 25] {
    keccak_p_parallel8::<24>(state)
}
