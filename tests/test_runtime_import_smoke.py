def test_runtime_package_exports_runtime_config():
    import rhecs_core.runtime as runtime

    assert hasattr(runtime, "RuntimeConfig")
