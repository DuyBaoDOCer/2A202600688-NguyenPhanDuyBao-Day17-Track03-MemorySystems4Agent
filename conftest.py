"""Root conftest.py — patches ssl before deepeval plugin loads on Windows."""
import ssl

_orig = ssl.SSLContext.load_default_certs


def _safe_load_default_certs(self, purpose=ssl.Purpose.SERVER_AUTH):  # type: ignore[override]
    try:
        _orig(self, purpose)
    except ssl.SSLError:
        pass


ssl.SSLContext.load_default_certs = _safe_load_default_certs  # type: ignore[method-assign]
