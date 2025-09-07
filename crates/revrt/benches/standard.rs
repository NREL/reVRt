//! Benchmarking reVRt
//!
//! Compare reVRt's performance to guide development and avoid regressions.
//!
//! Cases to consider:
//! - All ones: So we guarantee always the same solution
//! - Small distance but large cost array: It should be impacted by
//!   the cost chunk size only.
//! - Random cost: Would that create too much noise for statistics?
//! - Too many layers: How well we parallelize between layers.
//! - Too many paths in the same area: How well we parallelize
//!    between paths (re-using cost cache).
//! - Single chunk with reasonable size: How well we parallelize
//!   calculating the cost.

use core::time::Duration;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use revrt::ArrayIndex;
use revrt::bench_minimalist;

use ndarray::Array2;
use rand::Rng;

enum FeaturesType {
    AllOnes,
    Random,
}

fn features(ni: u64, nj: u64, ci: u64, cj: u64, ftype: FeaturesType) -> std::path::PathBuf {
    let tmp_path = tempfile::TempDir::new().unwrap();

    let store: zarrs::storage::ReadableWritableListableStorage = std::sync::Arc::new(
        zarrs::filesystem::FilesystemStore::new(tmp_path.path())
            .expect("could not open filesystem store"),
    );

    zarrs::group::GroupBuilder::new()
        .build(store.clone(), "/")
        .unwrap()
        .store_metadata()
        .unwrap();

    // Create an array
    // Remember to remove /cost
    for array_path in ["/A", "/B", "/C", "/cost"].iter() {
        let array = zarrs::array::ArrayBuilder::new(
            vec![ni, nj], // array shape
            zarrs::array::DataType::Float32,
            vec![ci, cj].try_into().unwrap(), // regular chunk shape
            zarrs::array::FillValue::from(zarrs::array::ZARR_NAN_F32),
        )
        // .bytes_to_bytes_codecs(vec![]) // uncompressed
        .dimension_names(["y", "x"].into())
        // .storage_transformers(vec![].into())
        .build(store.clone(), array_path)
        .unwrap();

        // Write array metadata to store
        array.store_metadata().unwrap();

        let mut a = vec![];
        match ftype {
            FeaturesType::AllOnes => {
                a.resize((ni * nj).try_into().unwrap(), 1.0);
            }
            FeaturesType::Random => {
                let mut rng = rand::rng();
                for _x in 0..(ni * nj) {
                    a.push(rng.random_range(0.0..=1.0));
                }
            }
        }
        let data: Array2<f32> =
            ndarray::Array::from_shape_vec((ni.try_into().unwrap(), nj.try_into().unwrap()), a)
                .unwrap();

        array
            .store_chunks_ndarray(
                &zarrs::array_subset::ArraySubset::new_with_ranges(&[0..(ni / ci), 0..(nj / cj)]),
                data,
            )
            .unwrap();
    }

    tmp_path.keep()
}

fn standard_ones(c: &mut Criterion) {
    let features_path = features(100, 100, 4, 4, FeaturesType::AllOnes);

    c.bench_function("constant_cost", |b| {
        b.iter(|| {
            bench_minimalist(
                black_box(features_path.clone()),
                black_box(vec![ArrayIndex::new(20, 50)]),
                black_box(vec![ArrayIndex::new(5, 50)]),
            )
        })
    });
}

fn standard_random(c: &mut Criterion) {
    let features_path = features(100, 100, 4, 4, FeaturesType::Random);

    c.bench_function("random_cost", |b| {
        b.iter(|| {
            bench_minimalist(
                black_box(features_path.clone()),
                black_box(vec![ArrayIndex::new(20, 50)]),
                black_box(vec![ArrayIndex::new(5, 50)]),
            )
        })
    });
}

fn single_chunk(c: &mut Criterion) {
    let features_path = features(100, 100, 1, 1, FeaturesType::AllOnes);

    c.bench_function("single_chunk", |b| {
        b.iter(|| {
            bench_minimalist(
                black_box(features_path.clone()),
                black_box(vec![ArrayIndex::new(20, 50)]),
                black_box(vec![ArrayIndex::new(5, 50)]),
            )
        })
    });
}

fn range_distance(c: &mut Criterion) {
    // Away from the border to progressively increas the search radius.
    static X0: u64 = 30;
    let features_path = features(100, 100, 1, 1, FeaturesType::AllOnes);

    let mut group = c.benchmark_group("distance");
    // Create an alternative benchmark definition to run locally only
    for distance in [0, 1, 2, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(distance),
            distance,
            |b, &distance| {
                b.iter(|| {
                    bench_minimalist(
                        black_box(features_path.clone()),
                        black_box(vec![ArrayIndex::new(X0 + distance, 50)]),
                        black_box(vec![ArrayIndex::new(X0, 50)]),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default().measurement_time(Duration::from_secs(25));
    targets = standard_ones, standard_random, single_chunk, range_distance
);
criterion_main!(benches);
