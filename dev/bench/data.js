window.BENCHMARK_DATA = {
  "lastUpdate": 1759681112410,
  "repoUrl": "https://github.com/NREL/reVRt",
  "entries": {
    "Rust Benchmark": [
      {
        "commit": {
          "author": {
            "email": "guilherme@castelao.net",
            "name": "Guilherme Castelão",
            "username": "castelao"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6209fec0b41fd6320d191ec7fb03b2a864884d76",
          "message": "Preserving benchmark report (#145)\n\n* Defining destination dir\n\n* Explicit target branch\n\n* doc: Updating documentation URL",
          "timestamp": "2025-10-05T10:09:55-06:00",
          "tree_id": "117443f8f8406c1a86b149614ffcaeb5d3faea2b",
          "url": "https://github.com/NREL/reVRt/commit/6209fec0b41fd6320d191ec7fb03b2a864884d76"
        },
        "date": 1759681111689,
        "tool": "cargo",
        "benches": [
          {
            "name": "constant_cost",
            "value": 74466303,
            "range": "± 749972",
            "unit": "ns/iter"
          },
          {
            "name": "random_cost",
            "value": 94560346,
            "range": "± 1102269",
            "unit": "ns/iter"
          },
          {
            "name": "single_chunk",
            "value": 434255208,
            "range": "± 16352362",
            "unit": "ns/iter"
          },
          {
            "name": "distance/0",
            "value": 142088593,
            "range": "± 765862",
            "unit": "ns/iter"
          },
          {
            "name": "distance/1",
            "value": 145315777,
            "range": "± 1294822",
            "unit": "ns/iter"
          },
          {
            "name": "distance/2",
            "value": 154147073,
            "range": "± 1052981",
            "unit": "ns/iter"
          },
          {
            "name": "distance/5",
            "value": 181892546,
            "range": "± 2875093",
            "unit": "ns/iter"
          },
          {
            "name": "distance/10",
            "value": 279867746,
            "range": "± 5283706",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}