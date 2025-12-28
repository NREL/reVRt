use std::sync::{Mutex, OnceLock};

use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::registry::Registry;
use tracing_subscriber::{fmt, layer::SubscriberExt, registry, reload, util::SubscriberInitExt};

use crate::error::{Error, Result};

type LogHandle = reload::Handle<LevelFilter, Registry>;
static LOG_HANDLE: OnceLock<Mutex<Option<LogHandle>>> = OnceLock::new();

pub(super) fn configure(level: Option<u8>) -> Result<()> {
    let tracing_level = match level {
        Some(level) => level,
        None => return Ok(()),
    };

    let tracing_level = match tracing_level {
        0..10 => tracing::Level::TRACE,
        10..20 => tracing::Level::DEBUG,
        20..30 => tracing::Level::INFO,
        30..40 => tracing::Level::WARN,
        _ => tracing::Level::ERROR,
    };

    let level_filter = LevelFilter::from_level(tracing_level);
    let handle_cell = LOG_HANDLE.get_or_init(|| Mutex::new(None));
    let mut guard = handle_cell
        .lock()
        .map_err(|err| Error::Undefined(format!("Failed to access logging state: {err}")))?;

    if let Some(handle) = guard.as_ref() {
        handle
            .reload(level_filter)
            .map_err(|err| Error::Undefined(format!("Failed to update log filter: {err}")))?;
        return Ok(());
    }

    let (reload_layer, handle) = reload::Layer::new(level_filter);
    registry()
        .with(reload_layer)
        .with(
            fmt::layer()
                .with_writer(std::io::stderr)
                .with_thread_ids(true),
        )
        .try_init()
        .map_err(|err| {
            Error::Undefined(format!("Failed to initialize tracing subscriber: {err}"))
        })?;

    *guard = Some(handle);
    Ok(())
}
