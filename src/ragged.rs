use ndarray::prelude::*;

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
