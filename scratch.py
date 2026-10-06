from synapsekit.sandbox.diff import DiffBundle

manifest = {
    "schema_version": "1.0",
    "host_root": "/tmp",
    "base_fingerprint": "test",
    "sandbox_id": "test",
    "changes": [
        {
            "kind": "add",
            "path": "file.txt",
            "size": "invalid_size",
            "mode": 0
        }
    ]
}

DiffBundle.from_dict(manifest)