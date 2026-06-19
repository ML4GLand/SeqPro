//! LUT gather for `seqpro.tokenize`: `out[i] = lut[seq[i]]` over a flat
//! `u8` buffer, choosing serial vs. rayon-parallel by an element-count
//! threshold. The 256-entry `i32` LUT is built in Python (NumPy).

use rayon::prelude::*;

/// Element count at or above which the rayon gather overtakes the serial one.
/// Re-measured for rayon in `benches/` (Task 5); the old Numba constant (40k)
/// was tuned to Numba's ~96µs thread-launch floor and does not transfer.
pub const TOKENIZE_PARALLEL_THRESHOLD: usize = 40_000;

/// Serial gather. `out` and `seq` must have equal length; `lut` has 256 entries.
pub fn gather_serial(seq: &[u8], lut: &[i32], out: &mut [i32]) {
    for (o, &s) in out.iter_mut().zip(seq.iter()) {
        *o = lut[s as usize];
    }
}

/// Parallel gather over contiguous slices.
pub fn gather_parallel(seq: &[u8], lut: &[i32], out: &mut [i32]) {
    out.par_iter_mut()
        .zip(seq.par_iter())
        .for_each(|(o, &s)| *o = lut[s as usize]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lut() -> Vec<i32> {
        let mut l = vec![-1i32; 256];
        l[b'A' as usize] = 0;
        l[b'C' as usize] = 1;
        l[b'G' as usize] = 2;
        l[b'T' as usize] = 3;
        l
    }

    #[test]
    fn serial_matches_expected() {
        let seq = b"ACGTN";
        let mut out = vec![0i32; seq.len()];
        gather_serial(seq, &lut(), &mut out);
        assert_eq!(out, vec![0, 1, 2, 3, -1]);
    }

    #[test]
    fn parallel_matches_serial() {
        let seq: Vec<u8> = (0..10_000u32).map(|i| b"ACGT"[(i % 4) as usize]).collect();
        let l = lut();
        let mut a = vec![0i32; seq.len()];
        let mut b = vec![0i32; seq.len()];
        gather_serial(&seq, &l, &mut a);
        gather_parallel(&seq, &l, &mut b);
        assert_eq!(a, b);
    }
}
