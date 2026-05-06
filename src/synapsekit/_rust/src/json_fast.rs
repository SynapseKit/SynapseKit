use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyNone, PyString};

/// Convert a Python object to a serde_json::Value entirely in Rust.
/// Handles dict, list, str, int, float, bool, None.
fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_instance_of::<PyNone>() {
        Ok(serde_json::Value::Null)
    } else if obj.is_instance_of::<PyBool>() {
        // Must check bool before int (bool is subclass of int in Python)
        let b: bool = obj.extract()?;
        Ok(serde_json::Value::Bool(b))
    } else if obj.is_instance_of::<PyInt>() {
        let i: i64 = obj.extract()?;
        Ok(serde_json::Value::Number(i.into()))
    } else if obj.is_instance_of::<PyFloat>() {
        let f: f64 = obj.extract()?;
        match serde_json::Number::from_f64(f) {
            Some(n) => Ok(serde_json::Value::Number(n)),
            None => Ok(serde_json::Value::Null), // NaN/Inf → null (matches json.dumps default)
        }
    } else if obj.is_instance_of::<PyString>() {
        let s: String = obj.extract()?;
        Ok(serde_json::Value::String(s))
    } else if obj.is_instance_of::<PyList>() {
        let list = obj.downcast::<PyList>()?;
        let mut arr = Vec::with_capacity(list.len());
        for item in list.iter() {
            arr.push(py_to_json(&item)?);
        }
        Ok(serde_json::Value::Array(arr))
    } else if obj.is_instance_of::<PyDict>() {
        let dict = obj.downcast::<PyDict>()?;
        let mut map = serde_json::Map::new();
        for (k, v) in dict.iter() {
            let key: String = k.extract()?;
            map.insert(key, py_to_json(&v)?);
        }
        Ok(serde_json::Value::Object(map))
    } else {
        // Fallback: convert via str()
        let s: String = obj.str()?.extract()?;
        Ok(serde_json::Value::String(s))
    }
}

/// Batch-serialize a list of metadata dicts to JSON strings.
///
/// Converts each Python dict to serde_json::Value in Rust, then serializes —
/// avoids repeated Python→C boundary crossings of stdlib json.dumps.
#[pyfunction]
pub fn serialize_metadata_list(metadata: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
    let mut result = Vec::with_capacity(metadata.len());
    for item in &metadata {
        let value = py_to_json(item)?;
        let s = serde_json::to_string(&value).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON serialization error: {e}"))
        })?;
        result.push(s);
    }
    Ok(result)
}

/// Batch-deserialize a list of JSON strings to Python dicts.
///
/// Parses JSON in Rust with serde_json, then converts to Python objects —
/// faster than calling json.loads N times through the Python interpreter.
#[pyfunction]
pub fn deserialize_metadata_list(py: Python<'_>, data: Vec<String>) -> PyResult<Vec<PyObject>> {
    let mut result = Vec::with_capacity(data.len());
    for s in &data {
        let value: serde_json::Value = serde_json::from_str(s).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {e}"))
        })?;
        let obj = json_to_py(py, &value)?;
        result.push(obj);
    }
    Ok(result)
}

/// Convert a serde_json::Value to a Python object.
fn json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<PyObject> {
    match value {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let list = PyList::empty(py);
            for item in arr {
                list.append(json_to_py(py, item)?)?;
            }
            Ok(list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, json_to_py(py, v)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_json_roundtrip() {
        let input = r#"{"name": "test", "value": 42, "nested": {"a": [1, 2, 3]}}"#;
        let parsed: serde_json::Value = serde_json::from_str(input).unwrap();
        let output = serde_json::to_string(&parsed).unwrap();
        let reparsed: serde_json::Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed, reparsed);
    }

    #[test]
    fn test_null_nan_inf() {
        // NaN and Inf should not appear in valid JSON
        assert!(serde_json::Number::from_f64(f64::NAN).is_none());
        assert!(serde_json::Number::from_f64(f64::INFINITY).is_none());
    }
}
