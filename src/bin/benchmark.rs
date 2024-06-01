#![cfg_attr(
    all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"),
    feature(stdarch_x86_avx512, avx512_target_feature)
)]

use std::hint::black_box;
use std::time::Instant;

use keccak::*;
const ITERS: usize = 10000000;

macro_rules! bench {
    ($n:literal, $keccak:ident, $interleave:ident, $fast_interleave:ident, $xor:ident) => {
        unsafe {
            let mut state = black_box($interleave(&[[0u64; 25]; $n]));
            let data = black_box([[0xbdu8; { 24 * 8 }]; $n]);

            for _ in 0..ITERS / 50 {
                for i in 0..24 / $n {
                    let input = $fast_interleave(data.map(|chunk| load(chunk.as_ptr().add(i * 8))));
                    for j in 0..$n {
                        state[i + j] = $xor(state[i + j], input[j]);
                    }
                }
                state = $keccak(&state);
            }

            let start = Instant::now();
            for _ in 0..ITERS {
                for i in 0..24 / $n {
                    let input = $fast_interleave(data.map(|chunk| load(chunk.as_ptr().add(8 * i))));
                    for j in 0..$n {
                        state[i + j] = $xor(state[i + j], input[j]);
                    }
                }
                state = $keccak(&state);
            }
            let end = Instant::now();

            println!(
                "x{} speed = {:.0} ns/iter",
                $n,
                (end - start).as_nanos() as f64 / ($n * ITERS) as f64
            );
        }
    };
    (1) => {
        let mut state = black_box([0u64; 25]);
        let data = black_box([0xbdu8; { 24 * 8 }]);

        for _ in 0..ITERS / 50 {
            for i in 0..24 {
                state[i] ^= u64::from_be_bytes(unsafe { *data.as_ptr().add(8 * i).cast() });
            }
            state = keccak_f(&state);
        }

        let start = Instant::now();
        for _ in 0..ITERS {
            for i in 0..24 {
                state[i] ^= u64::from_be_bytes(unsafe { *data.as_ptr().add(8 * i).cast() });
            }
            state = keccak_f(&state);
        }
        let end = Instant::now();

        println!(
            "Scalar speed = {:.0} ns/iter",
            (end - start).as_nanos() as f64 / ITERS as f64
        );
    };
    (2, $xor:ident) => {
        bench!(2, keccak_f_parallel2, interleave_state2, fast_interleave2, $xor);
    };
    (4, $xor:ident) => {
        bench!(4, keccak_f_parallel4, interleave_state4, fast_interleave4, $xor);
    };
    (8, $xor:ident) => {
        bench!(8, keccak_f_parallel8, interleave_state8, fast_interleave8, $xor);
    };
}

fn main() {
    bench!(1);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("sse2") {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        #[inline(always)]
        unsafe fn load(ptr: *const u8) -> __m128i {
            _mm_loadu_si128(ptr.cast())
        }
        bench!(2, _mm_xor_si128);
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if is_x86_feature_detected!("avx2") {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        #[inline(always)]
        unsafe fn load(ptr: *const u8) -> __m256i {
            _mm256_loadu_si256(ptr.cast())
        }
        bench!(4, _mm256_xor_si256);
    }
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "nightly"))]
    if is_x86_feature_detected!("avx512f") {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;
        #[inline(always)]
        unsafe fn load(ptr: *const u8) -> __m512i {
            _mm512_loadu_si512(ptr.cast())
        }
        bench!(8, _mm512_xor_si512);
    }
    #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
    if std::arch::is_aarch64_feature_detected!("neon") {
        use core::arch::aarch64::*;
        #[inline(always)]
        unsafe fn load(ptr: *const u8) -> uint64x2_t {
            vld1q_u64(ptr.cast())
        }
        bench!(2, veorq_u64);
    }
}
