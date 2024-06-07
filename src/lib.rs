mod keccak;
mod parallel_keccak;

pub use parallel_keccak::*;
pub use keccak::*;

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
    
    #[test]
    fn parallel_keccak_test() {
        let states = [[0; 25], [1; 25]];
        let mut parallel = ParallelKeccakState::from(states);
        parallel.keccak_f();
        let results = <[[u64; 25]; 2]>::from(parallel);
        assert_eq!(results[0], keccak_f(&states[0]));
        assert_eq!(results[1], keccak_f(&states[1]));
    }
}
