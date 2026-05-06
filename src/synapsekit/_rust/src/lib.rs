use pyo3::prelude::*;

mod chunker;
mod hash;
mod json_fast;

#[pymodule]
fn _rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunker::recursive_split, m)?)?;
    m.add_function(wrap_pyfunction!(chunker::character_split, m)?)?;
    m.add_function(wrap_pyfunction!(hash::fast_cache_key, m)?)?;
    m.add_function(wrap_pyfunction!(json_fast::serialize_metadata_list, m)?)?;
    m.add_function(wrap_pyfunction!(json_fast::deserialize_metadata_list, m)?)?;
    Ok(())
}
