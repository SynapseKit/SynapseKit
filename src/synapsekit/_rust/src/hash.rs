use pyo3::prelude::*;

/// Compute a deterministic cache key from request parameters.
///
/// Uses Python's `_json.dumps_bytes` to produce the canonical JSON bytes
/// (matching whatever backend is active: orjson or stdlib), then hashes
/// with xxh3_128 in Rust — single native call replaces Python-level
/// dict construction + JSON serialization + hash computation.
#[pyfunction]
pub fn fast_cache_key(
    model: &str,
    input: &Bound<'_, PyAny>,
    temperature: f64,
    max_tokens: i64,
) -> PyResult<String> {
    let py = input.py();

    // Build the dict in Python (matches insertion order of the Python code)
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("model", model)?;
    dict.set_item("input", input)?;
    dict.set_item("temperature", temperature)?;
    dict.set_item("max_tokens", max_tokens)?;

    // Use the same _json.dumps_bytes as the Python fallback path
    // This respects orjson if installed, ensuring identical cache keys
    let json_mod = py.import("synapsekit._json")?;
    let payload_bytes: Vec<u8> = json_mod
        .getattr("dumps_bytes")?
        .call1((dict,))?
        .extract()?;

    // Hash with xxh3_128 — matches Python xxhash path
    let hash = xxhash_rust::xxh3::xxh3_128(&payload_bytes);
    Ok(format!("{:032x}", hash))
}

#[cfg(test)]
mod tests {
    // Rust-only unit tests (no Python dependency)

    #[test]
    fn test_xxh3_deterministic() {
        let data = b"test data for hashing";
        let h1 = xxhash_rust::xxh3::xxh3_128(data);
        let h2 = xxhash_rust::xxh3::xxh3_128(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_xxh3_different_input() {
        let h1 = xxhash_rust::xxh3::xxh3_128(b"hello");
        let h2 = xxhash_rust::xxh3::xxh3_128(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_hex_format_length() {
        let hash = xxhash_rust::xxh3::xxh3_128(b"test");
        let hex = format!("{:032x}", hash);
        assert_eq!(hex.len(), 32);
    }
}
