use ndarray::prelude::*;

pub fn select(
    starts: ArrayView1<i64>,
    stops: ArrayView1<i64>,
    idx: ArrayView1<i64>,
) -> (Array1<i64>, Array1<i64>) {
    let s = idx.iter().map(|&i| starts[i as usize]).collect();
    let e = idx.iter().map(|&i| stops[i as usize]).collect();
    (Array1::from_vec(s), Array1::from_vec(e))
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
        let (s, e) = select(starts.view(), stops.view(), idx.view());
        assert_eq!(s, array![5i64, 0]);
        assert_eq!(e, array![10i64, 3]);
    }
    #[test]
    fn test_validate_rejects_nonmonotonic() {
        assert!(validate(array![0i64, 3, 2].view(), 10, 2).is_err());
    }
}
