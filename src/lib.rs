#![cfg_attr(
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"),
    feature(stdarch_x86_avx512, avx512_target_feature)
)]
#![cfg_attr(all(target_arch = "arm", feature = "nightly"), feature(stdarch_arm_neon_intrinsics))]
#![cfg_attr(all(target_arch = "wasm64", feature = "nightly"), feature(simd_wasm64))]
#![cfg_attr(
    all(target_arch = "loongarch64", feature = "nightly"),
    feature(stdarch_loongarch, loongarch_target_feature)
)]
#![allow(unused)] // todo: remove before merge

mod keccak;
pub mod parallel_keccak;

pub use keccak::*;

cfg_if::cfg_if! {
    if #[cfg(any(target_arch = "x86", target_arch = "x86_64"))] {
        mod keccakx2_x86;
        mod keccakx4_x86;
        cfg_if::cfg_if! {
            if #[cfg(feature = "nightly")] {
                mod keccakx8_x86;
                pub const MAX_PARALLELISM: usize = 8;
            } else {
                pub const MAX_PARALLELISM: usize = 4;
            }
        }
    } else if #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec", all(target_arch = "arm", feature = "nightly")))] {
        mod keccakx2_arm;
        pub const MAX_PARALLELISM: usize = 2;
    } else if #[cfg(any(target_arch = "wasm32", all(target_arch = "wasm64", feature = "nightly")))] {
        mod keccakx2_wasm;
        pub const MAX_PARALLELISM: usize = 2;
    } else if #[cfg(all(target_arch = "loongarch64", feature = "nightly"))] {
        mod keccakx2_la64;
        mod keccakx4_la64;
        pub const MAX_PARALLELISM: usize = 4;
    } else {
        pub const MAX_PARALLELISM: usize = 1;
    }
}

const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808a,
    0x8000000080008000,
    0x000000000000808b,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008a,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000a,
    0x000000008000808b,
    0x800000000000008b,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800a,
    0x800000008000000a,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
];

macro_rules! keccak_impl {
    ($state: expr, $dup: ident) => {
        assert!(ROUNDS <= 24);

        #[rustfmt::skip]
        let [
            mut s00, mut s01, mut s02, mut s03, mut s04,
            mut s05, mut s06, mut s07, mut s08, mut s09,
            mut s10, mut s11, mut s12, mut s13, mut s14,
            mut s15, mut s16, mut s17, mut s18, mut s19,
            mut s20, mut s21, mut s22, mut s23, mut s24
        ]  = $state;

        for &iota in &$crate::ROUND_CONSTANTS[24 - ROUNDS..] {
            let c0 = xor(xor(xor(s00, s05), s10), xor(s15, s20));
            let c1 = xor(xor(xor(s01, s06), s11), xor(s16, s21));
            let c2 = xor(xor(xor(s02, s07), s12), xor(s17, s22));
            let c3 = xor(xor(xor(s03, s08), s13), xor(s18, s23));
            let c4 = xor(xor(xor(s04, s09), s14), xor(s19, s24));

            let d0 = xor(c4, rotate_left!(c1, 1));
            let d1 = xor(c0, rotate_left!(c2, 1));
            let d2 = xor(c1, rotate_left!(c3, 1));
            let d3 = xor(c2, rotate_left!(c4, 1));
            let d4 = xor(c3, rotate_left!(c0, 1));

            let tmp = s01;
            s01 = rotate_left!(xor(s06, d1), 44);
            s06 = rotate_left!(xor(s09, d4), 20);
            s09 = rotate_left!(xor(s22, d2), 61);
            s22 = rotate_left!(xor(s14, d4), 39);
            s14 = rotate_left!(xor(s20, d0), 18);
            s20 = rotate_left!(xor(s02, d2), 62);
            s02 = rotate_left!(xor(s12, d2), 43);
            s12 = rotate_left!(xor(s13, d3), 25);
            s13 = rotate_left!(xor(s19, d4), 8);
            s19 = rotate_left!(xor(s23, d3), 56);
            s23 = rotate_left!(xor(s15, d0), 41);
            s15 = rotate_left!(xor(s04, d4), 27);
            s04 = rotate_left!(xor(s24, d4), 14);
            s24 = rotate_left!(xor(s21, d1), 2);
            s21 = rotate_left!(xor(s08, d3), 55);
            s08 = rotate_left!(xor(s16, d1), 45);
            s16 = rotate_left!(xor(s05, d0), 36);
            s05 = rotate_left!(xor(s03, d3), 28);
            s03 = rotate_left!(xor(s18, d3), 21);
            s18 = rotate_left!(xor(s17, d2), 15);
            s17 = rotate_left!(xor(s11, d1), 10);
            s11 = rotate_left!(xor(s07, d2), 6);
            s07 = rotate_left!(xor(s10, d0), 3);
            s10 = rotate_left!(xor(tmp, d1), 1);
            s00 = xor(s00, d0);

            let e0 = s00;
            let e1 = s01;

            s00 = chi(s00, s01, s02);
            s01 = chi(s01, s02, s03);
            s02 = chi(s02, s03, s04);
            s03 = chi(s03, s04, e0);
            s04 = chi(s04, e0, e1);

            let e0 = s05;
            let e1 = s06;

            s05 = chi(s05, s06, s07);
            s06 = chi(s06, s07, s08);
            s07 = chi(s07, s08, s09);
            s08 = chi(s08, s09, e0);
            s09 = chi(s09, e0, e1);

            let e0 = s10;
            let e1 = s11;

            s10 = chi(s10, s11, s12);
            s11 = chi(s11, s12, s13);
            s12 = chi(s12, s13, s14);
            s13 = chi(s13, s14, e0);
            s14 = chi(s14, e0, e1);

            let e0 = s15;
            let e1 = s16;

            s15 = chi(s15, s16, s17);
            s16 = chi(s16, s17, s18);
            s17 = chi(s17, s18, s19);
            s18 = chi(s18, s19, e0);
            s19 = chi(s19, e0, e1);

            let e0 = s20;
            let e1 = s21;

            s20 = chi(s20, s21, s22);
            s21 = chi(s21, s22, s23);
            s22 = chi(s22, s23, s24);
            s23 = chi(s23, s24, e0);
            s24 = chi(s24, e0, e1);

            s00 = xor(s00, $dup(iota as _));
        }

        #[rustfmt::skip]
        return [
            s00, s01, s02, s03, s04,
            s05, s06, s07, s08, s09,
            s10, s11, s12, s13, s14,
            s15, s16, s17, s18, s19,
            s20, s21, s22, s23, s24,
        ];
    };
}

use keccak_impl;

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn keccak_f_test() {
        let mut input = [0; 25];
        input = keccak_f(&input);
        #[rustfmt::skip]
        assert_eq!(input, [
            0xf1258f7940e1dde7, 0x84d5ccf933c0478a, 0xd598261ea65aa9ee, 0xbd1547306f80494d, 0x8b284e056253d057,
            0xff97a42d7f8e6fd4, 0x90fee5a0a44647c4, 0x8c5bda0cd6192e76, 0xad30a6f71b19059c, 0x30935ab7d08ffc64,
            0xeb5aa93f2317d635, 0xa9a6e6260d712103, 0x81a57c16dbcf555f, 0x43b831cd0347c826, 0x01f22f1a11a5569f,
            0x05e5635a21d9ae61, 0x64befef28cc970f2, 0x613670957bc46611, 0xb87c5a554fd00ecb, 0x8c3ee88a1ccf32c8,
            0x940c7922ae3a2614, 0x1841f924a2c509e4, 0x16f53526e70465c2, 0x75f644e97f30a13b, 0xeaf1ff7b5ceca249
        ]);
        input = keccak_f(&input);
        #[rustfmt::skip]
        assert_eq!(input, [
            0x2d5c954df96ecb3c, 0x6a332cd07057b56d, 0x093d8d1270d76b6c, 0x8a20d9b25569d094, 0x4f9c4f99e5e7f156,
            0xf957b9a2da65fb38, 0x85773dae1275af0d, 0xfaf4f247c3d810f7, 0x1f1b9ee6f79a8759, 0xe4fecc0fee98b425,
            0x68ce61b6b9ce68a1, 0xdeea66c4ba8f974f, 0x33c43d836eafb1f5, 0xe00654042719dbd9, 0x7cf8a9f009831265,
            0xfd5449a6bf174743, 0x97ddad33d8994b40, 0x48ead5fc5d0be774, 0xe3b8c8ee55b7b03c, 0x91a0226e649e42e9,
            0x900e3129e7badd7b, 0x202a9ec5faa3cce8, 0x5b3402464e1c3db6, 0x609f4e62a44c1059, 0x20d06cd26a8fbf5c
        ]);
    }
}
