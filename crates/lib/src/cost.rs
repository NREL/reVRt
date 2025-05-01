//! Cost fuction

use tracing::trace;

use crate::error::Result;

#[derive(Debug, serde::Deserialize)]
struct CostFunction {
    cost_layers: Vec<CostLayer>,
}

#[derive(Debug, serde::Deserialize)]
struct CostLayer {
    layer_name: String,
    multiplier_scalar: Option<f64>,
    multiplier_layer: Option<String>,
}

impl CostFunction {
    pub(super) fn from_json(json: &str) -> Result<Self> {
        trace!("Parsing cost definition from json: {}", json);
        let cost = serde_json::from_str(json).unwrap();
        Ok(cost)
    }
}

#[cfg(test)]
mod sample {

    /// Sample cost definition
    pub(crate) fn as_text_v1() -> String {
        r#"
      {
        "cost_layers": [
          {"layer_name": "layer_1_in_zarr"},
          {"layer_name": "layer_2_in_zarr", "multiplier_scalar": 100},
          {"layer_name": "layer_3_in_zarr", "multiplier_layer": "another_layer_in_zarr"},
          {"layer_name": "layer_4_in_zarr", "multiplier_layer": "another_layer_in_zarr", "multiplier_scalar": 2}
]
        }
        "#.to_string()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cost() {
        let json = sample::as_text_v1();
        let cost = CostFunction::from_json(&json).unwrap();

        assert_eq!(cost.cost_layers.len(), 4);
        assert_eq!(cost.cost_layers[0].layer_name, "layer_1_in_zarr");
        assert_eq!(cost.cost_layers[1].layer_name, "layer_2_in_zarr");
        assert_eq!(cost.cost_layers[1].multiplier_scalar, Some(100.0));
        assert_eq!(cost.cost_layers[2].layer_name, "layer_3_in_zarr");
        assert_eq!(
            cost.cost_layers[2].multiplier_layer,
            Some("another_layer_in_zarr".to_string())
        );
        assert_eq!(cost.cost_layers[3].layer_name, "layer_4_in_zarr");
        assert_eq!(
            cost.cost_layers[3].multiplier_layer,
            Some("another_layer_in_zarr".to_string())
        );
        assert_eq!(cost.cost_layers[3].multiplier_scalar, Some(2.0));
    }
}
