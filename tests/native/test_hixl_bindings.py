# SPDX-License-Identifier: Apache-2.0
"""Exercise the compiled production binding, not a Python copy of its errors."""

# Third Party
import pytest


def invoke(module, method):
    args = {
        "initialize": ("local", {}),
        "register_mem": (0x1000, 4096, module.MemType.MEM_HOST),
        "deregister_mem": (0x1234,),
        "connect": ("remote", 10000),
        "disconnect": ("remote", 1000),
        "transfer_sync": ("remote", module.TransferOp.READ, []),
        "transfer_async": ("remote", module.TransferOp.READ, []),
        "get_transfer_status": (0x5678,),
    }
    return getattr(module.Hixl(), method)(*args[method])


@pytest.mark.parametrize(
    "method,operation,holds_gil,code",
    [
        ("initialize", "Initialize", True, 103900),
        ("register_mem", "RegisterMem", False, 503900),
        ("deregister_mem", "DeregisterMem", False, 503900),
        ("connect", "Connect", False, 103903),
        ("disconnect", "Disconnect", False, 103902),
        # The shared translator must preserve unknown uint32_t codes too.
        ("transfer_sync", "TransferSync", False, 0xFFFFFFFF),
        ("transfer_async", "TransferAsync", False, 503900),
        ("get_transfer_status", "GetTransferStatus", False, 503900),
    ],
)
def test_native_errors_keep_code_operation_and_runtimeerror_compatibility(
    binding, method, operation, holds_gil, code
):
    module, control = binding
    control.set_result(code)
    with pytest.raises(RuntimeError) as caught:
        invoke(module, method)
    error = caught.value
    assert isinstance(error, module.HixlError)
    assert error.code == code
    assert error.operation == operation
    assert operation in str(error) and str(code) in str(error)
    assert control.calls() == [(operation, holds_gil)]


@pytest.mark.parametrize(
    "method,operation",
    [
        ("register_mem", "RegisterMem"),
        ("deregister_mem", "DeregisterMem"),
        ("transfer_async", "TransferAsync"),
        ("get_transfer_status", "GetTransferStatus"),
    ],
)
def test_success_returns_and_native_gil_boundaries(binding, method, operation):
    module, control = binding
    expected = {
        "register_mem": 0x1234,
        "transfer_async": 0x5678,
        "get_transfer_status": module.TransferStatus.COMPLETED,
    }.get(method)
    assert invoke(module, method) == expected
    assert control.calls() == [(operation, False)]


def test_query_error_takes_precedence_over_failed_output(binding):
    module, control = binding
    control.set_transfer_status(module.TransferStatus.FAILED)
    control.set_result(103902)
    with pytest.raises(module.HixlError) as caught:
        invoke(module, "get_transfer_status")
    assert caught.value.code == module.HIXL_NOT_CONNECTED
    control.set_result(0)
    assert invoke(module, "get_transfer_status") == module.TransferStatus.FAILED


def test_return_codes_are_not_shadowed_by_transfer_status_exports(binding):
    module, _ = binding
    # Old top-level enum exports stay usable. New callers compare raw codes.
    assert module.FAILED == module.TransferStatus.FAILED
    assert module.TIMEOUT == module.TransferStatus.TIMEOUT
    for name, code in [
        ("SUCCESS", 0),
        ("PARAM_INVALID", 103900),
        ("TIMEOUT", 103901),
        ("NOT_CONNECTED", 103902),
        ("ALREADY_CONNECTED", 103903),
        ("NOTIFY_FAILED", 103904),
        ("UNSUPPORTED", 103905),
        ("FAILED", 503900),
        ("RESOURCE_EXHAUSTED", 203900),
    ]:
        assert getattr(module, "HIXL_" + name) == code
        assert type(getattr(module, "HIXL_" + name)) is int


def test_unrelated_cpp_errors_are_not_hixl_errors(binding):
    module, control = binding
    control.throw_cpp_error()
    with pytest.raises(RuntimeError, match="external C\\+\\+ failure") as caught:
        invoke(module, "connect")
    assert not isinstance(caught.value, module.HixlError)
