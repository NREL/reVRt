//! Command line support for RevX-Transmission

use std::path::PathBuf;

use clap::Parser;
use tracing::{debug, info, trace};

use nrel_transmission::resolve;

#[derive(Parser)]
#[command(version, about, author, long_about = None)]
struct Cli {
    #[arg(short, long, action=clap::ArgAction::Count)]
    verbose: u8,

    #[arg(short, long, value_name = "DATASET")]
    dataset: PathBuf,

    #[arg(long = "cost-function", value_name = "COST_FUNCTION")]
    cost_function: String,

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
    debug!("Verbose level: {}", cli.verbose);

    trace!("User given dataset: {:?}", cli.dataset);

    assert_eq!(cli.start.len(), 2);
    let start = &nrel_transmission::Point::new(cli.start[0] as u64, cli.start[1] as u64);
    trace!("Starting point: {:?}", start);

    assert_eq!(cli.end.len(), 2);
    let end = vec![nrel_transmission::Point::new(cli.end[0] as u64, cli.end[1] as u64)];
    trace!("Ending point: {:?}", end);

    let result = resolve(
        cli.dataset,
        &cli.cost_function,
        250_000_000,
        &[start.clone()],
        end,
    )
    .unwrap();
    println!("Results: {:?}", result);
    info!("Final solutions: {:?}", result);
}
