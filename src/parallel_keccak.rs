// RUSTC is AMAZING at auto-vectorizing (it even uses SVE and RVV, which we cannot use from Rust (yet))
// just see asm to make sure its actually vectorized

use crate::ROUND_CONSTANTS;
use std::iter::zip;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ParallelKeccakState<const N: usize> {
    states: [[u64; N]; 25],
}

impl<const N: usize> From<[[u64; 25]; N]> for ParallelKeccakState<N> {
    fn from(src: [[u64; 25]; N]) -> Self {
        let mut dst = [[0; N]; 25];
        for lane in 0..25 {
            for p in 0..N {
                dst[lane][p] = src[p][lane];
            }
        }
        Self { states: dst }
    }
}

impl<const N: usize> From<ParallelKeccakState<N>> for [[u64; 25]; N] {
    fn from(src: ParallelKeccakState<N>) -> Self {
        let mut dst = [[0; 25]; N];
        for lane in 0..25 {
            for p in 0..N {
                dst[p][lane] = src.states[lane][p];
            }
        }
        dst
    }
}

impl<const N: usize> ParallelKeccakState<N> {
    pub fn xor_lane(&mut self, lane_idx: usize, lane: [u64; N]) {
        assert!(lane_idx < 25, "Keccak-p has a maximum of 25 lanes");
        for p in 0..N {
            self.states[lane_idx][p] ^= lane[p];
        }
    }

    pub fn xor_lanes<const LANES: usize>(&mut self, offset: usize, lanes: [[u64; LANES]; N]) {
        assert!(offset + LANES <= 25, "Keccak-p has a maximum of 25 lanes");
        for p in 0..N {
            for lane in 0..LANES {
                self.states[offset + lane][p] ^= lanes[p][lane];
            }
        }
    }

    pub unsafe fn load_and_xor<const LANES: usize>(
        &mut self,
        offset: usize,
        lanes: [*const u8; N],
    ) {
        assert!(offset + LANES <= 25, "Keccak-p has a maximum of 25 lanes");
        for p in 0..N {
            for lane in 0..LANES {
                self.states[offset + lane][p] ^= u64::from_le_bytes(*lanes[p].add(8 * lane).cast());
            }
        }
    }

    pub unsafe fn load_interleaved_and_xor<const LANES: usize>(
        &mut self,
        offset: usize,
        start: *const u8,
        interleaving: usize,
    ) {
        assert!(offset + LANES <= 25, "Keccak-p has a maximum of 25 lanes");
        for p in 0..N {
            for lane in 0..LANES {
                self.states[offset + lane][p] ^=
                    u64::from_le_bytes(*start.add(p * interleaving + 8 * lane).cast());
            }
        }
    }

    pub fn extract_lane(&self, lane_idx: usize) -> [u64; N] {
        assert!(lane_idx < 25, "Keccak-p has a maximum of 25 lanes");
        let mut dst = [0; N];
        for p in 0..N {
            dst[p] = self.states[lane_idx][p];
        }
        dst
    }

    pub fn extract_lanes<const LANES: usize>(&self, offset: usize) -> [[u64; LANES]; N] {
        assert!(offset + LANES <= 25, "Keccak-p has a maximum of 25 lanes");
        let mut dst = [[0; LANES]; N];
        for p in 0..N {
            for lane in 0..LANES {
                dst[p][lane] = self.states[offset + lane][p];
            }
        }
        dst
    }

    pub fn keccak_p(&mut self, rounds: usize) {
        assert!(rounds <= 24, "Keccak-p has a maximum of 24 rounds");
        for &iota in &ROUND_CONSTANTS[24 - rounds..] {
            #[rustfmt::skip]
            let [
                s00, s01, s02, s03, s04,
                s05, s06, s07, s08, s09,
                s10, s11, s12, s13, s14,
                s15, s16, s17, s18, s19,
                s20, s21, s22, s23, s24
            ]  = &mut self.states;

            for (
                (
                    (((s00, s01), (s02, s03)), ((s04, s05), (s06, s07))),
                    (((s08, s09), (s10, s11)), ((s12, s13), (s14, s15))),
                ),
                ((((s16, s17), (s18, s19)), ((s20, s21), (s22, s23))), s24),
            ) in zip(
                zip(
                    zip(zip(zip(s00, s01), zip(s02, s03)), zip(zip(s04, s05), zip(s06, s07))),
                    zip(zip(zip(s08, s09), zip(s10, s11)), zip(zip(s12, s13), zip(s14, s15))),
                ),
                zip(zip(zip(zip(s16, s17), zip(s18, s19)), zip(zip(s20, s21), zip(s22, s23))), s24),
            ) {
                let c0 = *s00 ^ *s05 ^ *s10 ^ *s15 ^ *s20;
                let c1 = *s01 ^ *s06 ^ *s11 ^ *s16 ^ *s21;
                let c2 = *s02 ^ *s07 ^ *s12 ^ *s17 ^ *s22;
                let c3 = *s03 ^ *s08 ^ *s13 ^ *s18 ^ *s23;
                let c4 = *s04 ^ *s09 ^ *s14 ^ *s19 ^ *s24;

                let d0 = c4 ^ c1.rotate_left(1);
                let d1 = c0 ^ c2.rotate_left(1);
                let d2 = c1 ^ c3.rotate_left(1);
                let d3 = c2 ^ c4.rotate_left(1);
                let d4 = c3 ^ c0.rotate_left(1);

                let tmp = *s01;
                *s01 = (*s06 ^ d1).rotate_left(44);
                *s06 = (*s09 ^ d4).rotate_left(20);
                *s09 = (*s22 ^ d2).rotate_left(61);
                *s22 = (*s14 ^ d4).rotate_left(39);
                *s14 = (*s20 ^ d0).rotate_left(18);
                *s20 = (*s02 ^ d2).rotate_left(62);
                *s02 = (*s12 ^ d2).rotate_left(43);
                *s12 = (*s13 ^ d3).rotate_left(25);
                *s13 = (*s19 ^ d4).rotate_left(8);
                *s19 = (*s23 ^ d3).rotate_left(56);
                *s23 = (*s15 ^ d0).rotate_left(41);
                *s15 = (*s04 ^ d4).rotate_left(27);
                *s04 = (*s24 ^ d4).rotate_left(14);
                *s24 = (*s21 ^ d1).rotate_left(2);
                *s21 = (*s08 ^ d3).rotate_left(55);
                *s08 = (*s16 ^ d1).rotate_left(45);
                *s16 = (*s05 ^ d0).rotate_left(36);
                *s05 = (*s03 ^ d3).rotate_left(28);
                *s03 = (*s18 ^ d3).rotate_left(21);
                *s18 = (*s17 ^ d2).rotate_left(15);
                *s17 = (*s11 ^ d1).rotate_left(10);
                *s11 = (*s07 ^ d2).rotate_left(6);
                *s07 = (*s10 ^ d0).rotate_left(3);
                *s10 = (tmp ^ d1).rotate_left(1);
                *s00 ^= d0;

                let e0 = *s00;
                let e1 = *s01;

                *s00 ^= !*s01 & *s02;
                *s01 ^= !*s02 & *s03;
                *s02 ^= !*s03 & *s04;
                *s03 ^= !*s04 & e0;
                *s04 ^= !e0 & e1;

                let e0 = *s05;
                let e1 = *s06;

                *s05 ^= !*s06 & *s07;
                *s06 ^= !*s07 & *s08;
                *s07 ^= !*s08 & *s09;
                *s08 ^= !*s09 & e0;
                *s09 ^= !e0 & e1;

                let e0 = *s10;
                let e1 = *s11;

                *s10 ^= !*s11 & *s12;
                *s11 ^= !*s12 & *s13;
                *s12 ^= !*s13 & *s14;
                *s13 ^= !*s14 & e0;
                *s14 ^= !e0 & e1;

                let e0 = *s15;
                let e1 = *s16;

                *s15 ^= !*s16 & *s17;
                *s16 ^= !*s17 & *s18;
                *s17 ^= !*s18 & *s19;
                *s18 ^= !*s19 & e0;
                *s19 ^= !e0 & e1;

                let e0 = *s20;
                let e1 = *s21;

                *s20 ^= !*s21 & *s22;
                *s21 ^= !*s22 & *s23;
                *s22 ^= !*s23 & *s24;
                *s23 ^= !*s24 & e0;
                *s24 ^= !e0 & e1;

                *s00 ^= iota;
            }
        }
    }

    pub fn keccak_f(&mut self) {
        self.keccak_p(24)
    }
}
