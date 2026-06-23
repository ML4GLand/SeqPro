use ndarray::prelude::*;

/// Borrowed, zero-copy view of a single-axis ragged array over a flat byte buffer.
/// `offsets` are in element units (length n_rows+1); `data` is the packed bytes;
/// `elem` is the byte size of one logical element.
pub struct Ragged<'a> {
    pub offsets: &'a [i64],
    pub data: &'a [u8],
    pub elem: usize,
}

impl<'a> Ragged<'a> {
    pub fn new(offsets: &'a [i64], data: &'a [u8], elem: usize) -> Self {
        Self {
            offsets,
            data,
            elem,
        }
    }

    pub fn n_rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn lengths(&self) -> Vec<i64> {
        self.offsets.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Validate monotonic offsets covering exactly `data.len()/elem` elements.
    pub fn validate(&self) -> Result<(), String> {
        if self.elem == 0 {
            return Err("elem must be positive".into());
        }
        let n_data = (self.data.len() / self.elem) as i64;
        validate(ArrayView1::from(self.offsets), n_data, self.n_rows() as i64)
    }

    /// Copy each row's first `min(row_len, out_len)` elements into a pre-filled
    /// `out` (flat uint8 view of a row-major (n_rows, out_len) buffer already
    /// filled with the pad value). Parallel across rows.
    pub fn to_padded_into(
        &self,
        out: &mut [u8],
        itemsize: usize,
        out_len: usize,
    ) -> Result<(), String> {
        use rayon::prelude::*;
        let n = self.n_rows();
        let row_stride = out_len * itemsize;
        if out.len() != n * row_stride {
            return Err(format!(
                "out has {} bytes, expected {}",
                out.len(),
                n * row_stride
            ));
        }
        if row_stride == 0 {
            return Ok(());
        }
        out.par_chunks_mut(row_stride)
            .enumerate()
            .for_each(|(i, dst)| {
                let row_len = (self.offsets[i + 1] - self.offsets[i]) as usize;
                let ncopy = row_len.min(out_len);
                let nbytes = ncopy * itemsize;
                let src = self.offsets[i] as usize * itemsize;
                dst[..nbytes].copy_from_slice(&self.data[src..src + nbytes]);
            });
        Ok(())
    }
}

/// Concatenate N ragged arrays along their ragged axis.
///
/// All inputs share the same number of groups G.  For each group g the
/// output contains the elements of input 0 followed by input 1, …, input N-1.
///
/// `data_list[i]` is the packed byte buffer for input i (length = n_elements_i * elem).
/// `offsets_list[i]` is the 1-D (G+1,) offset array for input i (element units, not bytes).
/// `elem` is the byte-size of one element (e.g. 4 for f32 or i32).
///
/// Returns `(out_data: Vec<u8>, out_offsets: Vec<i64>)` where `out_offsets` has
/// length G+1 and `out_data` has length `sum_i(offsets_i[G]) * elem` bytes.
pub fn ragged_concat(
    data_list: Vec<&[u8]>,
    offsets_list: Vec<&[i64]>,
    elem: usize,
) -> Result<(Vec<u8>, Vec<i64>), String> {
    use rayon::prelude::*;

    let n = data_list.len();
    if n == 0 {
        return Err("ragged_concat requires at least one input".into());
    }
    if offsets_list.len() != n {
        return Err("data_list and offsets_list must have the same length".into());
    }
    if elem == 0 {
        return Err("elem must be positive".into());
    }

    let g = offsets_list[0].len().saturating_sub(1); // number of groups
    for (i, off) in offsets_list.iter().enumerate() {
        if off.len() != g + 1 {
            return Err(format!(
                "offsets_list[{}] has length {} but expected {}",
                i,
                off.len(),
                g + 1
            ));
        }
    }

    // Validate and compute per-group output lengths.
    // out_lens[g] = sum_i (offsets_i[g+1] - offsets_i[g])
    let mut out_lens: Vec<usize> = Vec::with_capacity(g);
    let mut total_elems: usize = 0;
    for grp in 0..g {
        let mut grp_len: usize = 0;
        for i in 0..n {
            let a = offsets_list[i][grp];
            let b = offsets_list[i][grp + 1];
            if a < 0 || b < a {
                return Err(format!(
                    "input {}: invalid offsets at group {}: [{}, {})",
                    i, grp, a, b
                ));
            }
            let len = (b - a) as usize;
            // Validate byte span against data buffer
            let byte_b = (b as usize)
                .checked_mul(elem)
                .ok_or_else(|| format!("input {}: byte offset overflow at group {}", i, grp))?;
            if byte_b > data_list[i].len() {
                return Err(format!(
                    "input {}: data span out of bounds at group {}",
                    i, grp
                ));
            }
            grp_len = grp_len.checked_add(len).ok_or("output length overflow")?;
        }
        out_lens.push(grp_len);
        total_elems = total_elems
            .checked_add(grp_len)
            .ok_or("total length overflow")?;
    }

    // Build 1-D output offsets (cumulative sum of out_lens).
    let mut out_offsets: Vec<i64> = Vec::with_capacity(g + 1);
    out_offsets.push(0i64);
    for &l in &out_lens {
        let prev = *out_offsets.last().unwrap();
        out_offsets.push(prev + l as i64);
    }

    let total_bytes = total_elems
        .checked_mul(elem)
        .ok_or("output byte count overflow")?;
    let mut out_data: Vec<u8> = vec![0u8; total_bytes];

    // Strategy: rayon over groups — each group writes into a disjoint slice of
    // out_data, so no locking is required.
    // Safety: we split out_data into disjoint mutable slices per group.
    // Each group slice is of length out_lens[g]*elem bytes.
    // We use split_at_mut to peel off slices without overlapping.

    // Build per-group slices in a Vec.
    let mut group_slices: Vec<&mut [u8]> = Vec::with_capacity(g);
    {
        let mut rest: &mut [u8] = &mut out_data;
        for &grp_len in &out_lens {
            let chunk_bytes = grp_len * elem;
            let (head, tail) = rest.split_at_mut(chunk_bytes);
            group_slices.push(head);
            rest = tail;
        }
    }

    // Parallel copy: for each group, write inputs 0..n sequentially into the group slice.
    group_slices
        .par_iter_mut()
        .enumerate()
        .for_each(|(grp, grp_slice)| {
            let mut dst_pos = 0usize;
            for i in 0..n {
                let a = offsets_list[i][grp] as usize;
                let b = offsets_list[i][grp + 1] as usize;
                let src_byte_start = a * elem;
                let src_byte_end = b * elem;
                let src = &data_list[i][src_byte_start..src_byte_end];
                grp_slice[dst_pos..dst_pos + src.len()].copy_from_slice(src);
                dst_pos += src.len();
            }
        });

    Ok((out_data, out_offsets))
}

pub fn nested_gather(
    o0_starts: ArrayView1<i64>,
    o0_stops: ArrayView1<i64>,
    mask: ArrayView1<bool>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    let n_groups = o0_starts.len();
    if o0_stops.len() != n_groups {
        return Err("o0_starts and o0_stops length mismatch".into());
    }
    let mut counts = Vec::with_capacity(n_groups);
    let mut sel = Vec::new();
    for (&a, &b) in o0_starts.iter().zip(o0_stops.iter()) {
        if a < 0 || b < a {
            return Err("invalid o0 range".into());
        }
        let mut c = 0i64;
        for m in a..b {
            let mi = m as usize;
            if mi >= mask.len() {
                return Err("mask shorter than middle segment count".into());
            }
            if mask[mi] {
                sel.push(m);
                c += 1;
            }
        }
        counts.push(c);
    }
    Ok((Array1::from_vec(counts), Array1::from_vec(sel)))
}

pub fn select(
    starts: ArrayView1<i64>,
    stops: ArrayView1<i64>,
    idx: ArrayView1<i64>,
) -> Result<(Array1<i64>, Array1<i64>), String> {
    let n = starts.len();
    let mut s = Vec::with_capacity(idx.len());
    let mut e = Vec::with_capacity(idx.len());
    for &i in idx.iter() {
        if i < 0 || (i as usize) >= n {
            return Err(format!("index {} out of bounds for {} segments", i, n));
        }
        s.push(starts[i as usize]);
        e.push(stops[i as usize]);
    }
    Ok((Array1::from_vec(s), Array1::from_vec(e)))
}

pub fn validate(offsets: ArrayView1<i64>, n_data: i64, n_segments: i64) -> Result<(), String> {
    if offsets.len() as i64 - 1 != n_segments {
        return Err(format!(
            "segment count {} != {}",
            offsets.len() as i64 - 1,
            n_segments
        ));
    }
    let mut prev = i64::MIN;
    for &o in offsets.iter() {
        if o < prev {
            return Err("offsets must be monotonic".into());
        }
        if o < 0 || o > n_data {
            return Err("offset out of bounds".into());
        }
        prev = o;
    }
    Ok(())
}

type NestedPackOutput = Result<(Array1<i64>, Array1<i64>, Array1<u8>), String>;

pub fn nested_pack(
    o0_starts: ArrayView1<i64>,
    o0_stops: ArrayView1<i64>,
    o1_starts: ArrayView1<i64>,
    o1_stops: ArrayView1<i64>,
    src: ArrayView1<u8>,
    elem: i64,
) -> NestedPackOutput {
    if elem <= 0 {
        return Err("elem must be positive".into());
    }
    let n_groups = o0_starts.len();
    let mut o0 = Vec::with_capacity(n_groups + 1);
    o0.push(0i64);
    let mut o1 = vec![0i64];
    let mut out: Vec<u8> = Vec::new();
    let mut mid_count = 0i64;
    let src_slice = src.as_slice().ok_or("src must be contiguous")?;
    for (&a0, &b0) in o0_starts.iter().zip(o0_stops.iter()) {
        if a0 < 0 || b0 < a0 {
            return Err("invalid o0 range".into());
        }
        for m in a0..b0 {
            let mi = m as usize;
            if mi >= o1_starts.len() || mi >= o1_stops.len() {
                return Err("middle index out of bounds".into());
            }
            let a = o1_starts[mi]
                .checked_mul(elem)
                .ok_or("byte offset overflow")?;
            let b = o1_stops[mi]
                .checked_mul(elem)
                .ok_or("byte offset overflow")?;
            if a < 0 || b < a || b as usize > src_slice.len() {
                return Err("data span out of bounds".into());
            }
            out.extend_from_slice(&src_slice[a as usize..b as usize]);
            o1.push(out.len() as i64 / elem);
            mid_count += 1;
        }
        o0.push(mid_count);
    }
    Ok((
        Array1::from_vec(o0),
        Array1::from_vec(o1),
        Array1::from_vec(out),
    ))
}

/// Pack rows from `src` into caller-supplied `out` buffer.
///
/// `out` must have exactly `sum_i((stops[i] - starts[i]) * elem)` bytes.
/// Every byte in `out` is overwritten; the caller is responsible for
/// allocating `out` with the correct size (computed as
/// `sum_i((stops[i] - starts[i])) * elem`).
pub fn pack_into(
    starts: ArrayView1<i64>,
    stops: ArrayView1<i64>,
    src: ArrayView1<u8>,
    elem: i64,
    out: &mut [u8],
) -> Result<(), String> {
    use rayon::prelude::*;

    if elem <= 0 {
        return Err("elem must be positive".into());
    }
    let n = starts.len();
    if stops.len() != n {
        return Err("starts and stops length mismatch".into());
    }
    // Get raw slices for O(1) index access (avoids ndarray per-element overhead).
    let starts_s = starts.as_slice().ok_or("starts must be contiguous")?;
    let stops_s = stops.as_slice().ok_or("stops must be contiguous")?;
    let src_slice = src.as_slice().ok_or("src must be contiguous")?;

    // Up-front validation pass: check every row, build lens[], compute total.
    // This single pass covers BOTH the sequential and parallel paths so neither
    // can OOB-panic on a mis-sized `out`.
    let mut lens: Vec<usize> = Vec::with_capacity(n);
    let mut total_bytes: usize = 0;
    for (&a, &b) in starts_s.iter().zip(stops_s.iter()) {
        if a < 0 || b < a {
            return Err("invalid pack range".into());
        }
        let a_bytes = a.checked_mul(elem).ok_or("byte offset overflow")? as usize;
        let b_bytes = b.checked_mul(elem).ok_or("byte offset overflow")? as usize;
        if b_bytes > src_slice.len() {
            return Err("data span out of bounds".into());
        }
        lens.push(b_bytes - a_bytes);
        total_bytes += b_bytes - a_bytes;
    }
    if out.len() != total_bytes {
        return Err(format!(
            "out buffer has {} bytes but {} are required",
            out.len(),
            total_bytes
        ));
    }

    // Decide strategy based on output buffer size.
    // For small batches use a single-pass sequential gather (no extra allocation,
    // no rayon overhead).  For large batches chunk into ~num_cpus parallel tasks.
    const MIN_PAR_BYTES: usize = 4 * 1024 * 1024; // 4 MB
    let num_cpus = rayon::current_num_threads().max(1);

    if out.len() < MIN_PAR_BYTES || num_cpus == 1 {
        // Sequential gather: validation already done above, just copy rows.
        let mut dst_pos = 0usize;
        for (&a, &row_len) in starts_s.iter().zip(lens.iter()) {
            let a_bytes = (a * elem) as usize;
            out[dst_pos..dst_pos + row_len].copy_from_slice(&src_slice[a_bytes..a_bytes + row_len]);
            dst_pos += row_len;
        }
        return Ok(());
    }

    // Parallel path: lens[] and total_bytes already computed by the up-front pass.
    // Chunk rows into ~num_cpus groups by output byte count.
    let chunk_bytes = (total_bytes / num_cpus).max(1);
    let mut chunk_bounds: Vec<(usize, usize)> = Vec::with_capacity(num_cpus + 1);
    {
        let mut cumsum = 0usize;
        let mut chunk_start_row = 0usize;
        let mut chunk_start_bytes = 0usize;
        for (i, &row_len) in lens.iter().enumerate() {
            cumsum += row_len;
            if cumsum - chunk_start_bytes >= chunk_bytes && i + 1 < n {
                chunk_bounds.push((chunk_start_row, i + 1));
                chunk_start_row = i + 1;
                chunk_start_bytes = cumsum;
            }
        }
        if chunk_start_row < n {
            chunk_bounds.push((chunk_start_row, n));
        }
    }

    // Split output into one disjoint mutable sub-slice per chunk (safe).
    let mut chunk_slices: Vec<&mut [u8]> = Vec::with_capacity(chunk_bounds.len());
    {
        let mut rest = &mut *out;
        for &(first, last) in &chunk_bounds {
            let chunk_len: usize = lens[first..last].iter().sum();
            let (head, tail) = rest.split_at_mut(chunk_len);
            chunk_slices.push(head);
            rest = tail;
        }
    }

    // Parallel gather.
    chunk_slices
        .par_iter_mut()
        .zip(chunk_bounds.par_iter())
        .for_each(|(out_chunk, &(first, last))| {
            let mut dst_pos = 0usize;
            for (&a, &row_len) in starts_s[first..last].iter().zip(lens[first..last].iter()) {
                let src_start = (a * elem) as usize;
                out_chunk[dst_pos..dst_pos + row_len]
                    .copy_from_slice(&src_slice[src_start..src_start + row_len]);
                dst_pos += row_len;
            }
        });

    Ok(())
}

/// Allocating wrapper around `pack_into` — used by `#[cfg(test)]` only.
pub fn pack(
    starts: ArrayView1<i64>,
    stops: ArrayView1<i64>,
    src: ArrayView1<u8>,
    elem: i64,
) -> Result<Array1<u8>, String> {
    // Quick pre-pass to compute total output bytes so we can pre-allocate.
    // `pack_into` will re-validate, but tests don't care about the extra pass.
    if elem <= 0 {
        return Err("elem must be positive".into());
    }
    let starts_s = starts.as_slice().ok_or("starts must be contiguous")?;
    let stops_s = stops.as_slice().ok_or("stops must be contiguous")?;
    let src_len = src.len();
    let mut total = 0usize;
    for (&a, &b) in starts_s.iter().zip(stops_s.iter()) {
        if a < 0 || b < a {
            return Err("invalid pack range".into());
        }
        let a_b = a.checked_mul(elem).ok_or("byte offset overflow")? as usize;
        let b_b = b.checked_mul(elem).ok_or("byte offset overflow")? as usize;
        if b_b > src_len {
            return Err("data span out of bounds".into());
        }
        total += b_b - a_b;
    }
    let mut out = vec![0u8; total];
    pack_into(starts, stops, src, elem, &mut out)?;
    Ok(Array1::from_vec(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Naive reference: row-major copy with truncation into a pre-filled buffer.
    fn to_padded_ref(
        offsets: &[i64],
        data: &[u8],
        elem: usize,
        out_len: usize,
        fill: u8,
    ) -> Vec<u8> {
        let n = offsets.len() - 1;
        let mut out = vec![fill; n * out_len * elem];
        for i in 0..n {
            let row_len = (offsets[i + 1] - offsets[i]) as usize;
            let ncopy = row_len.min(out_len);
            let src = offsets[i] as usize * elem;
            let dst = i * out_len * elem;
            out[dst..dst + ncopy * elem].copy_from_slice(&data[src..src + ncopy * elem]);
        }
        out
    }

    proptest! {
        #[test]
        fn to_padded_matches_reference(
            rows in proptest::collection::vec(0usize..6, 1..8),
            elem in 1usize..4,
            out_len in 0usize..7,
        ) {
            let mut offsets = vec![0i64];
            for r in &rows { offsets.push(offsets.last().unwrap() + *r as i64); }
            let n_data = *offsets.last().unwrap() as usize;
            let data: Vec<u8> = (0..n_data * elem).map(|x| (x % 251) as u8).collect();

            let mut out = vec![0xAAu8; rows.len() * out_len * elem];
            Ragged::new(&offsets, &data, elem)
                .to_padded_into(&mut out, elem, out_len)
                .unwrap();

            let expected = to_padded_ref(&offsets, &data, elem, out_len, 0xAA);
            prop_assert_eq!(out, expected);
        }
    }

    #[test]
    fn test_nested_gather_selects_per_group() {
        let o0_starts = array![0i64, 2];
        let o0_stops = array![2i64, 4]; // group0: middles 0,1 ; group1: middles 2,3
        let mask = array![true, false, true, true]; // keep middle 0 (g0), 2,3 (g1)
        let (counts, idx) = nested_gather(o0_starts.view(), o0_stops.view(), mask.view()).unwrap();
        assert_eq!(counts, array![1i64, 2]);
        assert_eq!(idx, array![0i64, 2, 3]);
    }
    #[test]
    fn test_nested_gather_rejects_short_mask() {
        let o0_starts = array![0i64];
        let o0_stops = array![3i64];
        let mask = array![true, false]; // mask shorter than middle count
        assert!(nested_gather(o0_starts.view(), o0_stops.view(), mask.view()).is_err());
    }
    #[test]
    fn test_nested_gather_rejects_invalid_range() {
        let starts = array![3i64];
        let stops = array![1i64]; // b < a
        let mask = array![true, true, true, true];
        assert!(nested_gather(starts.view(), stops.view(), mask.view()).is_err());
    }
    #[test]
    fn test_select_gathers() {
        let starts = array![0i64, 3, 5];
        let stops = array![3i64, 5, 10];
        let idx = array![2i64, 0];
        let (s, e) = select(starts.view(), stops.view(), idx.view()).unwrap();
        assert_eq!(s, array![5i64, 0]);
        assert_eq!(e, array![10i64, 3]);
    }
    #[test]
    fn test_select_rejects_oob() {
        let starts = array![0i64, 3, 5];
        let stops = array![3i64, 5, 10];
        // index 3 is out of bounds for 3 segments
        let idx_oob = array![3i64];
        assert!(select(starts.view(), stops.view(), idx_oob.view()).is_err());
        // negative index is also out of bounds
        let idx_neg = array![-1i64];
        assert!(select(starts.view(), stops.view(), idx_neg.view()).is_err());
    }
    #[test]
    fn test_validate_rejects_nonmonotonic() {
        assert!(validate(array![0i64, 3, 2].view(), 10, 2).is_err());
    }
    #[test]
    fn test_nested_pack_two_level() {
        // group0: middles 0,1 ; group1: middle 2.  middle lens (elem=1): 2,1,3
        let o0_starts = array![0i64, 2];
        let o0_stops = array![2i64, 3];
        let o1_starts = array![0i64, 2, 3];
        let o1_stops = array![2i64, 3, 6];
        let src: Array1<u8> = array![10, 11, 20, 30, 31, 32];
        let (o0, o1, out) = nested_pack(
            o0_starts.view(),
            o0_stops.view(),
            o1_starts.view(),
            o1_stops.view(),
            src.view(),
            1,
        )
        .unwrap();
        assert_eq!(o0, array![0i64, 2, 3]);
        assert_eq!(o1, array![0i64, 2, 3, 6]);
        assert_eq!(out, array![10u8, 11, 20, 30, 31, 32]);
    }
    #[test]
    fn test_nested_pack_rejects_zero_elem() {
        let o0s = array![0i64];
        let o0e = array![1i64];
        let o1s = array![0i64];
        let o1e = array![2i64];
        let src = array![1u8, 2];
        assert!(nested_pack(
            o0s.view(),
            o0e.view(),
            o1s.view(),
            o1e.view(),
            src.view(),
            0
        )
        .is_err());
    }
    #[test]
    fn test_nested_pack_rejects_invalid_o0_range() {
        let o0s = array![3i64];
        let o0e = array![1i64]; // b0 < a0
        let o1s = array![0i64, 2];
        let o1e = array![2i64, 4];
        let src = array![1u8, 2, 3, 4];
        assert!(nested_pack(
            o0s.view(),
            o0e.view(),
            o1s.view(),
            o1e.view(),
            src.view(),
            1
        )
        .is_err());
    }

    // ── pack (single-level) tests ────────────────────────────────────────────

    #[test]
    fn test_pack_normal_multi_row() {
        // 3 rows of elem=1 bytes: src = [0,1,2,3,4,5,6,7,8,9]
        // row0: starts[0]=1, stops[0]=4  → bytes 1,2,3
        // row1: starts[1]=6, stops[1]=8  → bytes 6,7
        // row2: starts[2]=0, stops[2]=2  → bytes 0,1
        // expected packed: [1,2,3, 6,7, 0,1]
        let starts = array![1i64, 6, 0];
        let stops = array![4i64, 8, 2];
        let src: Array1<u8> = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let out = pack(starts.view(), stops.view(), src.view(), 1).unwrap();
        assert_eq!(out, array![1u8, 2, 3, 6, 7, 0, 1]);
    }

    #[test]
    fn test_pack_elem_gt1() {
        // elem=2: elements are pairs of bytes.  src has 4 elements (8 bytes).
        // row0: starts[0]=1, stops[0]=3 → elements 1..3 → bytes [2,3,4,5]
        // row1: starts[1]=0, stops[1]=1 → elements 0..1 → bytes [0,1]
        let starts = array![1i64, 0];
        let stops = array![3i64, 1];
        let src: Array1<u8> = array![0, 1, 2, 3, 4, 5, 6, 7];
        let out = pack(starts.view(), stops.view(), src.view(), 2).unwrap();
        assert_eq!(out, array![2u8, 3, 4, 5, 0, 1]);
    }

    #[test]
    fn test_pack_rejects_zero_elem() {
        let starts = array![0i64];
        let stops = array![2i64];
        let src: Array1<u8> = array![1, 2];
        assert!(pack(starts.view(), stops.view(), src.view(), 0).is_err());
    }

    #[test]
    fn test_pack_rejects_b_lt_a() {
        let starts = array![3i64];
        let stops = array![1i64]; // b < a
        let src: Array1<u8> = array![1, 2, 3, 4];
        assert!(pack(starts.view(), stops.view(), src.view(), 1).is_err());
    }

    #[test]
    fn test_pack_rejects_oob_span() {
        // starts[0]=0, stops[0]=5, but src only has 4 bytes with elem=1
        let starts = array![0i64];
        let stops = array![5i64];
        let src: Array1<u8> = array![1, 2, 3, 4];
        assert!(pack(starts.view(), stops.view(), src.view(), 1).is_err());
    }

    #[test]
    fn test_pack_permuted_starts_stops() {
        // Simulate a mask/slice gather: rows are NOT in source order.
        // src (elem=1) has 10 bytes: indices 0-9
        // row0: row 7..9 of source → bytes [7,8]
        // row1: row 2..5 of source → bytes [2,3,4]
        // row2: row 0..1 of source → bytes [0]
        // expected packed (same order as input): [7,8, 2,3,4, 0]
        let starts = array![7i64, 2, 0];
        let stops = array![9i64, 5, 1];
        let src: Array1<u8> = array![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let out = pack(starts.view(), stops.view(), src.view(), 1).unwrap();
        assert_eq!(out, array![7u8, 8, 2, 3, 4, 0]);
    }

    #[test]
    fn test_pack_parallel_path_matches_serial_permuted() {
        // Force the parallel path: construct input whose total output >= 4 MB.
        // 16 rows x 256 KiB each = 4 MiB, well above the 4 MB threshold.
        // Rows are in REVERSE source order (permuted) to exercise disjoint-chunk
        // gather correctness across chunk boundaries.
        const ROW_LEN: usize = 262_144; // 256 KiB
        const N_ROWS: usize = 16;
        const SRC_LEN: usize = ROW_LEN * N_ROWS; // 4 MiB

        // Deterministic source pattern for byte-level verification.
        let src_data: Vec<u8> = (0..SRC_LEN).map(|i| (i & 0xff) as u8).collect();
        let src = Array1::from(src_data.clone());

        // Row i reads source slice [(N_ROWS-1-i)*ROW_LEN .. (N_ROWS-i)*ROW_LEN].
        let starts: Array1<i64> = Array1::from(
            (0..N_ROWS)
                .rev()
                .map(|r| (r * ROW_LEN) as i64)
                .collect::<Vec<_>>(),
        );
        let stops: Array1<i64> = Array1::from(
            (0..N_ROWS)
                .rev()
                .map(|r| ((r + 1) * ROW_LEN) as i64)
                .collect::<Vec<_>>(),
        );

        // Serial reference: gather each row in order.
        let mut expected: Vec<u8> = Vec::with_capacity(SRC_LEN);
        for (&a, &b) in starts.iter().zip(stops.iter()) {
            expected.extend_from_slice(&src_data[a as usize..b as usize]);
        }

        let out = pack(starts.view(), stops.view(), src.view(), 1).unwrap();
        assert_eq!(out.as_slice().unwrap(), expected.as_slice());
    }
}
