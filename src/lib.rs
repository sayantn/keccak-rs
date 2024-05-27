#![cfg_attr(not(test), no_std)]
#![cfg_attr(
    all(feature = "nightly", any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx512f"),
    feature(stdarch_x86_avx512)
)]

use cfg_if::cfg_if;
use core::fmt::{Display, Formatter};
use core::ops::BitXorAssign;

cfg_if! {
    if #[cfg(all(
        feature = "nightly",
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx512f"
    ))] {
        mod keccak_avx512;
        pub use keccak_avx512::*;
    } else if #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    ))] {
        mod keccak_avx2;
        pub use keccak_avx2::*;
    } else {
        mod keccak_plain;
        pub use keccak_plain::*;
    }
}

impl KeccakState {
    pub fn keccak_p_inplace<const ROUNDS: usize>(&mut self) {
        *self = self.keccak_p::<ROUNDS>();
    }

    pub fn keccak_f(&self) -> Self {
        self.keccak_p::<24>()
    }

    pub fn keccak_f_inplace(&mut self) {
        self.keccak_p_inplace::<24>()
    }
}

impl<const LANES: usize> BitXorAssign<[u64; LANES]> for KeccakState {
    fn bitxor_assign(&mut self, rhs: [u64; LANES]) {
        *self = *self ^ rhs;
    }
}

impl Display for KeccakState {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        let state: [u64; 25] = (*self).into();
        writeln!(f, "{:016x} {:016x} {:016x} {:016x} {:016x}", state[0], state[1], state[2], state[3], state[4])?;
        writeln!(f, "{:016x} {:016x} {:016x} {:016x} {:016x}", state[5], state[6], state[7], state[8], state[9])?;
        writeln!(f, "{:016x} {:016x} {:016x} {:016x} {:016x}", state[10], state[11], state[12], state[13], state[14])?;
        writeln!(f, "{:016x} {:016x} {:016x} {:016x} {:016x}", state[15], state[16], state[17], state[18], state[19])?;
        writeln!(f, "{:016x} {:016x} {:016x} {:016x} {:016x}", state[20], state[21], state[22], state[23], state[24])?;
        Ok(())
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

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn keccak_f_test() {
        let mut input = KeccakState::from([0; 25]);
        input.keccak_f_inplace();
        #[rustfmt::skip]
        assert_eq!(input, [
            0xf1258f7940e1dde7, 0x84d5ccf933c0478a, 0xd598261ea65aa9ee, 0xbd1547306f80494d, 0x8b284e056253d057,
            0xff97a42d7f8e6fd4, 0x90fee5a0a44647c4, 0x8c5bda0cd6192e76, 0xad30a6f71b19059c, 0x30935ab7d08ffc64,
            0xeb5aa93f2317d635, 0xa9a6e6260d712103, 0x81a57c16dbcf555f, 0x43b831cd0347c826, 0x01f22f1a11a5569f,
            0x05e5635a21d9ae61, 0x64befef28cc970f2, 0x613670957bc46611, 0xb87c5a554fd00ecb, 0x8c3ee88a1ccf32c8,
            0x940c7922ae3a2614, 0x1841f924a2c509e4, 0x16f53526e70465c2, 0x75f644e97f30a13b, 0xeaf1ff7b5ceca249
        ].into());
        input.keccak_f_inplace();
        #[rustfmt::skip]
        assert_eq!(input, [
            0x2d5c954df96ecb3c, 0x6a332cd07057b56d, 0x093d8d1270d76b6c, 0x8a20d9b25569d094, 0x4f9c4f99e5e7f156,
            0xf957b9a2da65fb38, 0x85773dae1275af0d, 0xfaf4f247c3d810f7, 0x1f1b9ee6f79a8759, 0xe4fecc0fee98b425,
            0x68ce61b6b9ce68a1, 0xdeea66c4ba8f974f, 0x33c43d836eafb1f5, 0xe00654042719dbd9, 0x7cf8a9f009831265,
            0xfd5449a6bf174743, 0x97ddad33d8994b40, 0x48ead5fc5d0be774, 0xe3b8c8ee55b7b03c, 0x91a0226e649e42e9,
            0x900e3129e7badd7b, 0x202a9ec5faa3cce8, 0x5b3402464e1c3db6, 0x609f4e62a44c1059, 0x20d06cd26a8fbf5c
        ].into());
    }

    #[test]
    fn xor_test() {
        let mut state = KeccakState::from([0; 25]);
        state ^= [1, 2, 3, 4, 5, 6];
        #[rustfmt::skip]
        assert_eq!(state, [
            1, 2, 3, 4, 5,
            6, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ].into());
        state ^= [1, 2, 3];
        #[rustfmt::skip]
        assert_eq!(state, [
            0, 0, 0, 4, 5,
            6, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ].into());
    }
}
