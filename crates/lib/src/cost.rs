//! Cost fuction

use tracing::trace;

use crate::error::Result;

#[derive(Debug, serde::Deserialize)]
pub(crate) struct CostFunction {
    cost_layers: Vec<CostLayer>,
}

#[derive(Debug, serde::Deserialize)]
struct CostLayer {
    layer_name: String,
    multiplier_scalar: Option<f32>,
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
          {"layer_name": "A"},
          {"layer_name": "B", "multiplier_scalar": 100},
          {"layer_name": "A",
            "multiplier_layer": "B"},
          {"layer_name": "C",
            "multiplier_layer": "A",
            "multiplier_scalar": 2}
]
        }
        "#
        .to_string()
    }

    pub(crate) fn cost_function() -> CostFunction {
        let json = as_text_v1();
        CostFunction::from_json(&json).unwrap()
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
        assert_eq!(cost.cost_layers[0].layer_name, "A");
        assert_eq!(cost.cost_layers[1].layer_name, "B");
        assert_eq!(cost.cost_layers[1].multiplier_scalar, Some(100.0));
        assert_eq!(cost.cost_layers[2].layer_name, "A");
        assert_eq!(
            cost.cost_layers[2].multiplier_layer,
            Some("B".to_string())
        );
        assert_eq!(cost.cost_layers[3].layer_name, "C");
        assert_eq!(
            cost.cost_layers[3].multiplier_layer,
            Some("A".to_string())
        );
        assert_eq!(cost.cost_layers[3].multiplier_scalar, Some(2.0));
    }
}
