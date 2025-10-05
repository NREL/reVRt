struct Simulation {
    scenario: Scenario,
    algorithm: Algorithm,
}

impl Simulation {
    fn new(scenario: Scenario, algorithm: Algorithm) -> Self {
        Self {
            scenario,
            algorithm,
        }
    }

    fn compute(&self) {}
}

struct Solution {}

struct Scenario {
    features: Features,
    cost_function: CostFunction,
}

struct Features {}
struct CostFunction {}

struct Algorithm {}
