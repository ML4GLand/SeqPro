//! Rolling integer encoder for k-mers over an arbitrary byte alphabet.
//!
//! The encoder maps each (k-1)-byte window of a sequence to a `u32` integer
//! in `0..alphabet_size.pow(k-1)`. A 256-entry byte-to-code table allows
//! arbitrary alphabets (DNA = ACGT, RNA = ACGU, protein, etc.).

/// Build a 256-entry table mapping ASCII byte → alphabet code.
/// Bytes not present in `alphabet_bytes` map to `u8::MAX` (sentinel; callers
/// are expected to have sanitized input — current code does not check either).
pub fn build_byte_to_code(alphabet_bytes: &[u8]) -> [u8; 256] {
    let mut table = [u8::MAX; 256];
    for (code, &b) in alphabet_bytes.iter().enumerate() {
        debug_assert!(code < 256);
        table[b as usize] = code as u8;
    }
    table
}

/// Encode the first (k-1)-mer of `seq` as a base-`alphabet_size` integer.
/// Most-significant digit is `seq[0]`. Panics if `seq.len() < k_minus_1`.
pub fn encode_first(seq: &[u8], k_minus_1: usize, alphabet_size: u32, b2c: &[u8; 256]) -> u32 {
    let mut code: u32 = 0;
    for &b in &seq[..k_minus_1] {
        code = code * alphabet_size + b2c[b as usize] as u32;
    }
    code
}

/// Rolling update: given previous (k-1)-mer code at position `p`, return
/// the code at position `p+1`. `base_pow_km2 = alphabet_size^(k-2)` (precomputed).
/// `drop_byte = seq[p]`, `add_byte = seq[p + k - 1]`.
#[inline]
pub fn roll(
    prev: u32,
    drop_byte: u8,
    add_byte: u8,
    base_pow_km2: u32,
    alphabet_size: u32,
    b2c: &[u8; 256],
) -> u32 {
    let drop_code = b2c[drop_byte as usize] as u32;
    let add_code = b2c[add_byte as usize] as u32;
    (prev - drop_code * base_pow_km2) * alphabet_size + add_code
}

#[cfg(test)]
mod tests {
    use super::*;

    const DNA: &[u8] = b"ACGT";

    #[test]
    fn byte_to_code_dna() {
        let b2c = build_byte_to_code(DNA);
        assert_eq!(b2c[b'A' as usize], 0);
        assert_eq!(b2c[b'C' as usize], 1);
        assert_eq!(b2c[b'G' as usize], 2);
        assert_eq!(b2c[b'T' as usize], 3);
        assert_eq!(b2c[b'N' as usize], u8::MAX);
    }

    #[test]
    fn encode_first_acgt_k4() {
        // (k-1) = 3, alphabet=4: ACG -> 0*16 + 1*4 + 2 = 6
        let b2c = build_byte_to_code(DNA);
        assert_eq!(encode_first(b"ACGT", 3, 4, &b2c), 0 * 16 + 1 * 4 + 2);
    }

    #[test]
    fn rolling_matches_recomputed() {
        // For sequence ACGTACGT and k-1 = 3:
        // windows: ACG, CGT, GTA, TAC, ACG, CGT
        let b2c = build_byte_to_code(DNA);
        let seq = b"ACGTACGT";
        let k_minus_1 = 3;
        let alpha = 4u32;
        let base_pow_km2 = alpha.pow((k_minus_1 - 1) as u32); // 4^2 = 16

        let mut prev = encode_first(seq, k_minus_1, alpha, &b2c);
        for i in 0..(seq.len() - k_minus_1) {
            let recomputed = encode_first(&seq[i..], k_minus_1, alpha, &b2c);
            assert_eq!(prev, recomputed, "mismatch at window {}", i);
            if i + 1 + k_minus_1 <= seq.len() {
                prev = roll(prev, seq[i], seq[i + k_minus_1], base_pow_km2, alpha, &b2c);
            }
        }
    }
}
