use crate::ROUND_CONSTANTS;

pub fn keccak_p<const ROUNDS: usize>(dst: &mut [u64; 25], src: &[u64; 25]) {
    assert!(ROUNDS <= 24);
    
    #[rustfmt::skip]
    let [
        mut s00, mut s01, mut s02, mut s03, mut s04,
        mut s05, mut s06, mut s07, mut s08, mut s09,
        mut s10, mut s11, mut s12, mut s13, mut s14,
        mut s15, mut s16, mut s17, mut s18, mut s19,
        mut s20, mut s21, mut s22, mut s23, mut s24
    ] = *src;

    for i in 24 - ROUNDS..24 {
        let c0 = s00 ^ s05 ^ s10 ^ s15 ^ s20;
        let c1 = s01 ^ s06 ^ s11 ^ s16 ^ s21;
        let c2 = s02 ^ s07 ^ s12 ^ s17 ^ s22;
        let c3 = s03 ^ s08 ^ s13 ^ s18 ^ s23;
        let c4 = s04 ^ s09 ^ s14 ^ s19 ^ s24;

        let d0 = c4 ^ c1.rotate_left(1);
        let d1 = c0 ^ c2.rotate_left(1);
        let d2 = c1 ^ c3.rotate_left(1);
        let d3 = c2 ^ c4.rotate_left(1);
        let d4 = c3 ^ c0.rotate_left(1);

        let tmp = s01;
        s01 = (s06 ^ d1).rotate_left(44);
        s06 = (s09 ^ d4).rotate_left(20);
        s09 = (s22 ^ d2).rotate_left(61);
        s22 = (s14 ^ d4).rotate_left(39);
        s14 = (s20 ^ d0).rotate_left(18);
        s20 = (s02 ^ d2).rotate_left(62);
        s02 = (s12 ^ d2).rotate_left(43);
        s12 = (s13 ^ d3).rotate_left(25);
        s13 = (s19 ^ d4).rotate_left(8);
        s19 = (s23 ^ d3).rotate_left(56);
        s23 = (s15 ^ d0).rotate_left(41);
        s15 = (s04 ^ d4).rotate_left(27);
        s04 = (s24 ^ d4).rotate_left(14);
        s24 = (s21 ^ d1).rotate_left(2);
        s21 = (s08 ^ d3).rotate_left(55);
        s08 = (s16 ^ d1).rotate_left(45);
        s16 = (s05 ^ d0).rotate_left(36);
        s05 = (s03 ^ d3).rotate_left(28);
        s03 = (s18 ^ d3).rotate_left(21);
        s18 = (s17 ^ d2).rotate_left(15);
        s17 = (s11 ^ d1).rotate_left(10);
        s11 = (s07 ^ d2).rotate_left(6);
        s07 = (s10 ^ d0).rotate_left(3);
        s10 = (tmp ^ d1).rotate_left(1);
        s00 ^= d0;

        let e0 = s00;
        let e1 = s01;

        s00 ^= !s01 & s02;
        s01 ^= !s02 & s03;
        s02 ^= !s03 & s04;
        s03 ^= !s04 & e0;
        s04 ^= !e0 & e1;

        let e0 = s05;
        let e1 = s06;

        s05 ^= !s06 & s07;
        s06 ^= !s07 & s08;
        s07 ^= !s08 & s09;
        s08 ^= !s09 & e0;
        s09 ^= !e0 & e1;

        let e0 = s10;
        let e1 = s11;

        s10 ^= !s11 & s12;
        s11 ^= !s12 & s13;
        s12 ^= !s13 & s14;
        s13 ^= !s14 & e0;
        s14 ^= !e0 & e1;

        let e0 = s15;
        let e1 = s16;

        s15 ^= !s16 & s17;
        s16 ^= !s17 & s18;
        s17 ^= !s18 & s19;
        s18 ^= !s19 & e0;
        s19 ^= !e0 & e1;

        let e0 = s20;
        let e1 = s21;

        s20 ^= !s21 & s22;
        s21 ^= !s22 & s23;
        s22 ^= !s23 & s24;
        s23 ^= !s24 & e0;
        s24 ^= !e0 & e1;

        s00 ^= ROUND_CONSTANTS[i];
    }

    //temp variable for attribute
    #[rustfmt::skip]
    let ret = [
        s00, s01, s02, s03, s04,
        s05, s06, s07, s08, s09,
        s10, s11, s12, s13, s14,
        s15, s16, s17, s18, s19,
        s20, s21, s22, s23, s24,
    ];

    *dst = ret;
}
