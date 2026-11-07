# SPDX-License-Identifier: Apache-2.0
"""Check pure ACL allocation and legacy mapped allocation call paths."""

# Standard
from pathlib import Path
import os
import subprocess
import sys

# Third Party
import pytest


def test_pure_alloc_and_free_use_only_acl_host_calls_with_gil(host_alloc):
    ptr = host_alloc.alloc_host_ptr(size=12345)
    assert ptr == 0x12340000
    assert host_alloc.free_host_ptr(ptr=ptr) is None
    assert host_alloc.calls() == [
        ("aclrtMallocHost", 12345, True),
        ("aclrtFreeHost", 0x12340000, True),
    ]


def test_zero_size_is_rejected_before_acl(host_alloc):
    with pytest.raises(ValueError):
        host_alloc.alloc_host_ptr(0)
    assert host_alloc.calls() == []


def test_alloc_error_preserves_operation_code_size_without_cleanup(host_alloc):
    host_alloc.fail_alloc(123456)
    with pytest.raises(RuntimeError) as caught:
        host_alloc.alloc_host_ptr(12345)
    assert type(caught.value) is RuntimeError
    for text in ("aclrtMallocHost", "123456", "12345"):
        assert text in str(caught.value)
    assert host_alloc.calls() == [("aclrtMallocHost", 12345, True)]


def test_free_error_is_reported_once_with_original_pointer(host_alloc):
    ptr = host_alloc.alloc_host_ptr(4096)
    host_alloc.fail_free(654321)
    with pytest.raises(RuntimeError) as caught:
        host_alloc.free_host_ptr(ptr)
    assert type(caught.value) is RuntimeError
    assert "aclrtFreeHost" in str(caught.value)
    assert "654321" in str(caught.value)
    assert str(ptr) in str(caught.value) or hex(ptr) in str(caught.value)
    assert host_alloc.calls() == [
        ("aclrtMallocHost", 4096, True),
        ("aclrtFreeHost", ptr, True),
    ]


def test_legacy_pinned_path_still_maps_and_unmaps(host_alloc):
    ptr = host_alloc.alloc_pinned_ptr(4096, 0)
    host_alloc.free_pinned_ptr(ptr)
    assert host_alloc.calls() == [
        ("aclrtMallocHost", 4096, True),
        ("aclrtGetSocName", 0, True),
        ("register_ptr", ptr, True),
        ("unregister_ptr", ptr, True),
        ("aclrtFreeHost", ptr, True),
    ]


def test_missing_hixl_headers_skip_only_hixl_tests():
    pytest.importorskip("pybind11")
    environment = os.environ.copy()
    environment.pop("HIXL_INCLUDE_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--confcutdir=tests/native",
            "-q",
            "--tb=short",
            "tests/native/test_hixl_bindings.py::"
            "test_query_error_takes_precedence_over_failed_output",
            "tests/native/test_host_alloc.py::"
            "test_pure_alloc_and_free_use_only_acl_host_calls_with_gil",
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 skipped" in result.stdout
