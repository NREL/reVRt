//! Cost fuction

use derive_builder::Builder;
use ndarray::{Axis, stack};
use tracing::{debug, trace};

use crate::dataset::LazySubset;
use crate::error::Result;

#[derive(Debug, serde::Deserialize)]
/// A cost function definition
///
/// `cost_layers`: A collection of cost layers with equal weight.
///
/// This was based on the original transmission router and is composed of
/// layers that are summed together (per grid point) to give the total cost.
pub(crate) struct CostFunction {
    cost_layers: Vec<CostLayer>,
}

#[derive(Builder, Clone, Debug, serde::Deserialize)]
/// A cost layer
///
/// Each cost layer is a raster dataset, i.e. a regular grid, composed by
/// operating on input features. Following the original `revX` structure,
/// the possible compositions are limited to combinations of the relation
/// `weight * layer_name * multiplier_layer`, where the `weight` and the
/// `multiplier_layer` are optional.
struct CostLayer {
    layer_name: String,
    #[builder(setter(strip_option), default)]
    multiplier_scalar: Option<f32>,
    #[builder(setter(strip_option, into), default)]
    multiplier_layer: Option<String>,
}

impl CostFunction {
    /// Create a new cost function from a JSON string (reVX format)
    ///
    /// # Arguments
    /// `json`: A JSON string representing the cost function with the format
    ///         used by reVX.
    ///
    /// # Returns
    /// A `CostFunction` object.
    ///
    /// The JSON pattern used by reVX was the following:
    /// ```json
    /// {"cost_layers": [
    ///   {"layer_name": "A"},
    ///   {"layer_name": "A", "multiplier_scalar": 2, "multiplier_layer": "B"}
    ///   ]}
    /// ```
    pub(super) fn from_json(json: &str) -> Result<Self> {
        trace!("Parsing cost definition from json: {}", json);
        let cost = serde_json::from_str(json).unwrap();
        Ok(cost)
    }

    /// Calculate the cost from a given collection of input features
    ///
    /// Applies the cost function to a collection of input features, which
    /// is typically a subset of a larger dataset, such as a chunk from a
    /// Zarr dataset. The cost function is defined by a series of layers,
    /// each of which may have a multiplier scalar or a multiplier layer.
    ///
    /// # Arguments
    /// `features`: A lazy collection of input features.
    ///
    /// # Returns
    /// A 2D array containing the cost for the subset covered by the input
    /// features.
    pub(crate) fn compute(
        &self,
        mut features: LazySubset<f32>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>> {
        debug!("Calculating cost for ({})", features.subset());

        let cost = self
            .cost_layers
            .iter()
            .map(|layer| {
                let layer_name = &layer.layer_name;
                trace!("Layer name: {}", layer_name);

                let mut cost = features
                    .get(layer_name)
                    .expect("Layer not found in features");

                if let Some(multiplier_scalar) = layer.multiplier_scalar {
                    trace!(
                        "Layer {} has multiplier scalar {}",
                        layer_name, multiplier_scalar
                    );
                    // Apply the multiplier scalar to the value
                    cost *= multiplier_scalar;
                    // trace!( "Cost for chunk ({}, {}) in layer {}: {}", ci, cj, layer_name, cost);
                }

                if let Some(multiplier_layer) = &layer.multiplier_layer {
                    trace!(
                        "Layer {} has multiplier layer {}",
                        layer_name, multiplier_layer
                    );
                    let multiplier_value = features
                        .get(multiplier_layer)
                        .expect("Multiplier layer not found in features");

                    // Apply the multiplier layer to the value
                    cost = cost * multiplier_value;
                    // trace!( "Cost for chunk ({}, {}) in layer {}: {}", ci, cj, layer_name, cost);
                }
                cost
            })
            .collect::<Vec<_>>();

        let views: Vec<_> = cost.iter().map(|a| a.view()).collect();
        let stack = stack(Axis(0), &views).unwrap();
        //let cost = stack![Axis(3), &cost];
        trace!("Stack shape: {:?}", stack.shape());
        let cost = stack.sum_axis(Axis(0));
        trace!("Stack shape: {:?}", stack.shape());

        cost
    }
}

#[cfg(test)]
pub(crate) mod sample {
    use super::*;

    /// Sample cost definition
    pub(crate) fn as_text_v1() -> String {
        r#"
        {
            "cost_layers": [
                {"layer_name": "A"},
                {"layer_name": "B", "multiplier_scalar": 100},
                {"layer_name": "A",
                    "multiplier_layer": "B"},
                {"layer_name": "C", "multiplier_scalar": 2,
                    "multiplier_layer": "A"}
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
mod test_builder {
    use super::*;

    #[test]
    fn costlayer() {
        let layer = CostLayerBuilder::default()
            .layer_name("A".to_string())
            .multiplier_scalar(2.0)
            .multiplier_layer("B")
            .build()
            .unwrap();

        assert_eq!(layer.layer_name, "A");
        assert_eq!(layer.multiplier_scalar, Some(2.0));
        assert_eq!(layer.multiplier_layer, Some("B".to_string()));
    }

    #[test]
    fn defaults() {
        let layer = CostLayerBuilder::default()
            .layer_name("A".to_string())
            .build()
            .unwrap();

        assert_eq!(layer.layer_name, "A");
        assert_eq!(layer.multiplier_scalar, None);
        assert_eq!(layer.multiplier_layer, None);
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
        assert_eq!(cost.cost_layers[2].multiplier_layer, Some("B".to_string()));
        assert_eq!(cost.cost_layers[3].layer_name, "C");
        assert_eq!(cost.cost_layers[3].multiplier_layer, Some("A".to_string()));
        assert_eq!(cost.cost_layers[3].multiplier_scalar, Some(2.0));
    }
}
