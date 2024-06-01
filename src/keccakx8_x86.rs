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
pub unsafe fn keccak_p_parallel8<const ROUNDS: usize>(state: &[__m512i; 25]) -> [__m512i; 25] {
    crate::keccak_impl!(state, _mm512_set1_epi64);
}
