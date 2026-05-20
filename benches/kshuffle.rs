use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ndarray::Array2;
use seqpro::kshuffle::k_shuffle;

fn make_batch(n_rows: usize, len: usize) -> Array2<u8> {
    let mut arr = Array2::<u8>::zeros((n_rows, len));
    for (i, mut row) in arr.rows_mut().into_iter().enumerate() {
        for (j, b) in row.iter_mut().enumerate() {
            *b = b"ACGT"[((i.wrapping_mul(7919) + j.wrapping_mul(31)) ^ (i + j)) % 4];
        }
    }
    arr
}

fn bench_kshuffle(c: &mut Criterion) {
    let mut group = c.benchmark_group("k_shuffle/10k_x_200bp");
    let batch = make_batch(10_000, 200);
    for &k in &[2usize, 4, 6, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, &k| {
            b.iter(|| {
                let out = k_shuffle(black_box(batch.view()), k, Some(42), 4, b"ACGT");
                black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_kshuffle);
criterion_main!(benches);
