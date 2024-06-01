use std::ops::BitXor;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ParallelKeccakState<const PAR: usize> {
    states: [[u64; PAR]; 25],
}

impl<const PAR: usize> From<[[u64; 25]; PAR]> for ParallelKeccakState<PAR> {
    fn from(value: [[u64; 25]; PAR]) -> Self {
        todo!()
    }
}

impl<const PAR: usize> From<ParallelKeccakState<PAR>> for [[u64; 25]; PAR] {
    fn from(value: ParallelKeccakState<PAR>) -> Self {
        todo!()
    }
}

impl<const PAR: usize> ParallelKeccakState<PAR> {
    pub fn xor_lane<const LANE: usize>(&mut self, lane: [u64; PAR]) {
        todo!()
    }

    pub fn xor_lanes<const LANES: usize>(&mut self, lanes: [[u64; PAR]; LANES]) {
        todo!()
    }

    pub unsafe fn load_and_xor<const LANES: usize>(&mut self, lanes: [*const u8; LANES]) {
        todo!()
    }

    pub fn keccak_p<const ROUNDS: usize>(&self) -> Self {
        todo!()
    }
}
