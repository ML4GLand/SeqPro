use ndarray::prelude::*;

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

#[cfg(test)]
mod tests {
    use super::*;
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
}
