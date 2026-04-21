//! TQ1 v4 row decoder — base-3 packing, 5 ternaries per byte.
//!
//! Matches `rocm-cpp/kernels/ternary_gemv_tq1_halo.hip`'s LUT:
//!
//! ```text
//!   byte = d0 + d1 * 3 + d2 * 9 + d3 * 27 + d4 * 81
//!   d_i ∈ {0,1,2} → ternary ∈ {-1, 0, +1}
//! ```
//!
//! TQ1's `cols_padded` rounds the logical `cols` up to a multiple of 20 (so
//! every macro-group of 20 weights is exactly 4 bytes). Padding digits on
//! the writer side are `digit = 1` (i.e. ternary `0`), matching
//! `pack_tq1_v4` in `bench_sherry.cpp`.
//!
//! This module is read-only: no writer. The requantizer's write-side uses
//! `sherry_pack::pack_sherry_row` against the unpacked i8 buffer.

/// Logical cols rounded up to a multiple of 20 (TQ1's macro-group size).
/// Matches `H1bWeightFormat::TQ1V4.row_bytes(cols)` — always returns the
/// padded size in bytes.
pub fn tq1_row_bytes(cols: usize) -> usize {
    let cols_padded = cols.div_ceil(20) * 20;
    cols_padded / 5
}

/// Unpack one TQ1-v4 row of `cols` logical weights (not padded cols) into
/// `out`. `packed` must be exactly `tq1_row_bytes(cols)` bytes. `out` must
/// be exactly `cols` elements — padding weights (if any) are discarded.
///
/// Each byte decodes to five `i8` ternary values. Byte values ≥ 243 are
/// undefined in-spec; this decoder treats them as all-zero which matches
/// the kernel's LUT fill behaviour (`tq1_lut_*` are zero-filled above 242).
pub fn unpack_tq1_row(packed: &[u8], out: &mut [i8], cols: usize) {
    assert_eq!(
        packed.len(),
        tq1_row_bytes(cols),
        "tq1_unpack: packed.len mismatch",
    );
    assert_eq!(out.len(), cols, "tq1_unpack: out.len mismatch");

    let cols_padded = cols.div_ceil(20) * 20;
    let mut k = 0usize; // position in padded weight stream
    for &byte in packed {
        let mut b = byte as u32;
        let mut digits = [0i8; 5];
        if b < 243 {
            for slot in digits.iter_mut() {
                let d = (b % 3) as i8;
                *slot = d - 1; // {-1, 0, +1}
                b /= 3;
            }
        }
        // else: all five digits stay 0 (matches kernel LUT fallback).

        for slot in digits.iter() {
            if k < cols {
                out[k] = *slot;
            }
            // k == cols..cols_padded are padding — writer put zeros there.
            k += 1;
            if k >= cols_padded {
                break;
            }
        }
        if k >= cols_padded {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a TQ1 row of 20 ternaries (4 bytes) where only the first byte
    /// is set; the other three are 0 (which decodes to five `-1`s each).
    fn row20_with_leading_byte(byte: u8) -> [u8; 4] {
        [byte, 0, 0, 0]
    }

    #[test]
    fn round_trip_leading_byte() {
        // Byte = d0 + d1*3 + d2*9 + d3*27 + d4*81
        //      = 2 + 0*3 + 1*9 + 2*27 + 0*81 = 65
        //      → digits [2,0,1,2,0] → ternary [+1, -1, 0, +1, -1]
        let byte: u8 = 2 + 1 * 9 + 2 * 27;
        assert_eq!(byte, 65);
        let row = row20_with_leading_byte(byte);
        let mut out = [0i8; 20];
        unpack_tq1_row(&row, &mut out, 20);
        assert_eq!(out[..5], [1, -1, 0, 1, -1]);
        // The remaining 15 weights come from three zero bytes → all -1.
        assert_eq!(out[5..], [-1i8; 15]);
    }

    #[test]
    fn all_zero_byte_decodes_all_minus_one() {
        let row = row20_with_leading_byte(0);
        let mut out = [0i8; 20];
        unpack_tq1_row(&row, &mut out, 20);
        assert_eq!(out, [-1i8; 20]);
    }

    #[test]
    fn padding_respects_cols_cap() {
        // cols = 18 (not multiple of 20) still works: tq1_row_bytes rounds
        // up so packed is 4 bytes, but we only write 18 out-slots.
        let row = row20_with_leading_byte(0);
        let mut out = [0i8; 18];
        unpack_tq1_row(&row, &mut out, 18);
        assert_eq!(out, [-1i8; 18]);
    }

    #[test]
    fn oversize_byte_decodes_zero() {
        // byte = 250 (> 242) → all five slots stay 0 (kernel LUT fallback).
        let row = row20_with_leading_byte(250);
        let mut out = [0i8; 20];
        unpack_tq1_row(&row, &mut out, 20);
        // First 5 = zero (from the oversize byte), last 15 = -1.
        assert_eq!(out[..5], [0, 0, 0, 0, 0]);
        assert_eq!(out[5..], [-1i8; 15]);
    }
}
