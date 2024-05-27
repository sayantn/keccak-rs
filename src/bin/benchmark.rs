use keccak::KeccakState;
use std::env;
use std::hint::black_box;
use std::num::ParseIntError;
use std::str::FromStr;
use std::time::Instant;

fn main() -> Result<(), ParseIntError> {
    let iters = usize::from_str(&env::args().nth(1).unwrap())?;

    let mut state = black_box(KeccakState::from([0u64; 25]));
    let data = black_box([1; 21]);

    let start = Instant::now();
    for _ in 0..iters {
        state ^= data;
        state.keccak_f_inplace();
    }
    let end = Instant::now();

    println!("Speed: {:.2} iters/s", iters as f64 / (end - start).as_secs_f64());

    Ok(())
}
