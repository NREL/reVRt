//! Command line support for RevX-Transmission

use clap::Parser;
use tracing_subscriber;

#[derive(Parser)]
#[command(version, about, author, long_about = None)]
struct Cli {
    #[arg(short, long, action=clap::ArgAction::Count)]
    verbose: u8,
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
    tracing::info!("Verbose level: {}", cli.verbose);
}
