//! Defining build script to be executed before compiling

/// Adds linker arguments suitable for PyO3’s extension-module feature.
fn main() {
    pyo3_build_config::add_extension_module_link_args();
}
