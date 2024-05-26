use crate::ROUND_CONSTANTS;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::mem;

#[rustfmt::skip]
const LEFT_SHIFTS: [__m256i; 6] = unsafe { mem::transmute([
    1_u64, 62, 28, 27,
    3, 18, 36, 41,
    45, 6, 56, 39,
    10, 61, 55, 8,
    2, 15, 25, 20,
    44, 43, 21, 14
]) };

#[rustfmt::skip]
const RIGHT_SHIFTS:[__m256i; 6] = unsafe { mem::transmute([
    63_u64, 2, 36, 37,
    61, 46, 28, 23,
    19, 58, 8, 25,
    54, 3, 9, 56,
    62, 49, 39, 44,
    20, 21, 43, 50
]) };

#[inline(always)]
unsafe fn load(a: &[u64]) -> __m256i {
    _mm256_loadu_si256(a.as_ptr().cast())
}

#[inline(always)]
unsafe fn store(a: &mut [u64], b: __m256i) {
    _mm256_storeu_si256(a.as_mut_ptr().cast(), b)
}

#[inline(always)]
unsafe fn xor(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(a, b)
}

#[inline(always)]
unsafe fn rolv(a: __m256i, idx: usize) -> __m256i {
    _mm256_or_si256(
        _mm256_sllv_epi64(a, LEFT_SHIFTS[idx]),
        _mm256_srlv_epi64(a, RIGHT_SHIFTS[idx]),
    )
}

#[inline(always)]
unsafe fn rol1(a: __m256i) -> __m256i {
    _mm256_or_si256(_mm256_slli_epi64::<1>(a), _mm256_srli_epi64::<63>(a))
}

#[inline(always)]
unsafe fn perm<const PERM: i32>(a: __m256i) -> __m256i {
    _mm256_permute4x64_epi64::<PERM>(a)
}

#[inline(always)]
unsafe fn blend<const BLEND: i32>(a: __m256i, b: __m256i) -> __m256i {
    _mm256_blend_epi32::<BLEND>(a, b)
}

#[inline(always)]
unsafe fn and_not(a: __m256i, b: __m256i) -> __m256i {
    _mm256_andnot_si256(a, b)
}

#[cfg(test)]
unsafe fn print_state(
    stage: &str,
    s0: __m256i,
    s1: __m256i,
    s2: __m256i,
    s3: __m256i,
    s4: __m256i,
    s5: __m256i,
    s6: __m256i,
) {
    fn print(x: i64) {
        print!("{x:016x} ");
    }
    println!("{stage}: ");
    print(_mm256_extract_epi64::<0>(s0));
    print(_mm256_extract_epi64::<0>(s1));
    print(_mm256_extract_epi64::<1>(s1));
    print(_mm256_extract_epi64::<2>(s1));
    print(_mm256_extract_epi64::<3>(s1));
    println!();
    print(_mm256_extract_epi64::<0>(s2));
    print(_mm256_extract_epi64::<1>(s2));
    print(_mm256_extract_epi64::<2>(s2));
    print(_mm256_extract_epi64::<3>(s2));
    print(_mm256_extract_epi64::<0>(s3));
    println!();
    print(_mm256_extract_epi64::<1>(s3));
    print(_mm256_extract_epi64::<2>(s3));
    print(_mm256_extract_epi64::<3>(s3));
    print(_mm256_extract_epi64::<0>(s4));
    print(_mm256_extract_epi64::<1>(s4));
    println!();
    print(_mm256_extract_epi64::<2>(s4));
    print(_mm256_extract_epi64::<3>(s4));
    print(_mm256_extract_epi64::<0>(s5));
    print(_mm256_extract_epi64::<1>(s5));
    print(_mm256_extract_epi64::<2>(s5));
    println!();
    print(_mm256_extract_epi64::<3>(s5));
    print(_mm256_extract_epi64::<0>(s6));
    print(_mm256_extract_epi64::<1>(s6));
    print(_mm256_extract_epi64::<2>(s6));
    print(_mm256_extract_epi64::<3>(s6));
    println!();
}

pub fn keccak_p<const ROUNDS: usize>(dst: &mut [u64], src: &[u64]) {
    unsafe {
        let mut s0 = _mm256_set1_epi64x(src[0] as i64);
        let mut s1 = load(&src[1..]);
        let mut s2 = load(&src[5..]);
        let mut s3 = load(&src[9..]);
        let mut s4 = load(&src[13..]);
        let mut s5 = load(&src[17..]);
        let mut s6 = load(&src[21..]);

        for i in 24 - ROUNDS..24 {
            let t12 = xor(xor(xor(s1, s3), s4), xor(s5, s6));
            let t11 = perm::<0x93>(t12);
            let t13 = xor(s2, _mm256_shuffle_epi32::<0x4e>(s2));

            let t8 = rol1(t12);

            let t15 = perm::<0x39>(t8);
            let t14 = perm::<0>(xor(t11, t8));

            let t13 = xor(xor(t13, s0), perm::<0x4e>(t13));
            let t8 = rol1(t13);

            let t15 = xor(blend::<0x03>(t11, t13), blend::<0xc0>(t15, t8));

            s0 = xor(t14, s0);
            s1 = xor(t15, s1);
            let t9 = rolv(s1, 0);
            s2 = rolv(xor(t14, s2), 1);
            s3 = rolv(xor(t15, s3), 2);
            s4 = rolv(xor(t15, s4), 3);
            s5 = rolv(xor(t15, s5), 4);
            s6 = xor(t15, s6);
            let t8 = rolv(s6, 5);

            let t10 = perm::<0x8d>(s2);
            let t11 = perm::<0x8d>(s3);
            let t12 = perm::<0x1b>(s4);
            let t13 = perm::<0x72>(s5);

            let t7 = and_not(t8, _mm256_bsrli_epi128::<8>(t8));
            s1 = and_not(
                blend::<0xc0>(perm::<0x39>(t8), s0),
                blend::<0x30>(perm::<0x1e>(t8), s0),
            );
            s2 = and_not(
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t11, t12), t13), t10),
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t13, t11), t10), t12),
            );
            s3 = and_not(
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t9, t13), t11), t12),
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t11, t9), t12), t13),
            );
            s4 = and_not(
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t13, t10), t12), t9),
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t12, t13), t9), t10),
            );
            s5 = and_not(
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t10, t11), t9), t13),
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t9, t10), t13), t11),
            );
            s6 = and_not(
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t12, t9), t10), t11),
                blend::<0xc0>(blend::<0x30>(blend::<0x0c>(t10, t12), t11), t9),
            );

            s0 = perm::<0>(xor(t7, s0));
            s1 = xor(t8, s1);
            s2 = xor(t9, s2);
            s3 = perm::<0x1b>(xor(t10, s3));
            s4 = xor(t11, s4);
            s5 = perm::<0x8d>(xor(t12, s5));
            s6 = perm::<0x72>(xor(t13, s6));

            // Iota

            s0 = xor(s0, _mm256_set1_epi64x(ROUND_CONSTANTS[i] as i64));

            #[cfg(test)]
            print_state(&format!("Round-{i}: After Iota: "), s0, s1, s2, s3, s4, s5, s6);
        }

        _mm_storel_epi64(dst.as_mut_ptr().cast(), _mm256_castsi256_si128(s0));
        store(&mut dst[1..], s1);
        store(&mut dst[5..], s2);
        store(&mut dst[9..], s3);
        store(&mut dst[13..], s4);
        store(&mut dst[17..], s5);
        store(&mut dst[21..], s6);
    }
}
