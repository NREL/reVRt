use revrt::resolve;
use std::path::PathBuf;

const TEST_DATA: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/data/utilities/transmission_layers.zarr"
);

#[test]
fn basic_routing_in_data() {
    let layers_path = PathBuf::from(TEST_DATA);
    let start = &revrt::ArrayIndex::new(10, 10);
    let end = vec![revrt::ArrayIndex::new(20, 20)];
    let result = resolve(
        layers_path.to_str().expect("test data path is valid UTF-8"),
        r#"{
            "cost_layers": [{"layer_name": "tie_line_costs_102MW"}],
            "ignore_null_costs": true
        }"#,
        250_000_000,
        std::slice::from_ref(start),
        end,
    )
    .unwrap();
    dbg!(&result);
    assert_eq!(result.len(), 1);
    assert!(result[0].0.len() > 1);
    assert!(result[0].1 > 0.);
}
