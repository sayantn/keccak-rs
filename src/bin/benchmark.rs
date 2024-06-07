use std::hint::black_box;
use std::time::Instant;

use keccak::parallel_keccak::ParallelKeccakState;
use keccak::*;

const ITERS: usize = 50000;

macro_rules! bench_parallel {
    ($($p:literal),*) => {$(
        let mut state = black_box([[0_u64; 25]; $p]);
        let mut data = black_box([[0x07_u64; 21]; $p]);

        for _ in 0..ITERS / 50 {
            let mut parallel = ParallelKeccakState::from(state);
            for _ in 0..50 {
                parallel.xor_lanes(0, black_box(data));
                parallel.keccak_f();
            }
            data = black_box(parallel.extract_lanes(4));
            state = black_box(parallel.into());
        }

        let start = Instant::now();
        for _ in 0..ITERS {
            let mut parallel = ParallelKeccakState::from(state);
            for _ in 0..50 {
                parallel.xor_lanes(0, black_box(data));
                parallel.keccak_f();
            }
            data = black_box(parallel.extract_lanes(4));
            state = black_box(parallel.into());
        }
        let end = Instant::now();

        println!("x{} speed: {} ns/chunk", $p, (end - start).as_nanos() as f64 / ($p * ITERS) as f64);
    )*};
}

fn main() {
    let mut state = black_box([0; 25]);
    let mut data = black_box([0x07; 21]);

    for _ in 0..ITERS / 50 {
        for _ in 0..50 {
            for i in 0..21 {
                state[i] ^= black_box(data[i]);
            }
            state = keccak_f(&state);
        }
        for i in 0..21 {
            data[i] = black_box(state[i + 4]);
        }
    }

    let start = Instant::now();
    for _ in 0..ITERS {
        for _ in 0..50 {
            for i in 0..21 {
                state[i] ^= black_box(data[i]);
            }
            state = keccak_f(&state);
        }
        for i in 0..21 {
            data[i] = black_box(state[i + 4]);
        }
    }
    let end = Instant::now();

    println!("serial speed: {} ns/chunk", (end - start).as_nanos() as f64 / ITERS as f64);
    
    bench_parallel!(1, 2, 4, 8, 16);
}
