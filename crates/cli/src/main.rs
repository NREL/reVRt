//! Command line support for RevX-Transmission

use std::path::PathBuf;

use clap::Parser;
use tracing::{info, trace};
use tracing_subscriber;

use nrel_transmission::Simulation;

#[derive(Parser)]
#[command(version, about, author, long_about = None)]
struct Cli {
    #[arg(short, long, action=clap::ArgAction::Count)]
    verbose: u8,

    #[arg(short, long, value_name = "DATASET")]
    dataset: PathBuf,

    #[arg(long = "start", value_delimiter = ',', value_name = "START")]
    start: Vec<usize>,

    #[arg(long = "end", value_delimiter = ',', value_name = "END")]
    end: Vec<usize>,

    #[arg(long = "cache-size", value_name = "CACHE_SIZE")]
    cache_size: Option<usize>,
}

fn main() {
    let cli = Cli::parse();

    let tracing_level = match cli.verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    };
    tracing_subscriber::fmt()
        .with_max_level(tracing_level)
        .init();
    info!("Verbose level: {}", cli.verbose);

    trace!("Loading dataset: {:?}", cli.dataset);
    let mut simulation = Simulation::new(cli.dataset, 250_000_000).unwrap();

    assert_eq!(cli.start.len(), 2);
    let start = &nrel_transmission::Point(cli.start[0] as u64, cli.start[1] as u64);
    trace!("Starting point: {:?}", start);

    assert_eq!(cli.end.len(), 2);
    let end = vec![nrel_transmission::Point(
        cli.end[0] as u64,
        cli.end[1] as u64,
    )];
    trace!("Ending point: {:?}", end);

    let result = simulation.scout(&[start.clone()], end);
    println!("Results: {:?}", result);
    info!("Final solutions: {:?}", result);
}
