from ._version import (  # noqa: F401 imported but unused
    __version__,
    version_info,
)


def get_include(user=False):
    import os
    d = os.path.dirname(__file__)
    if os.path.exists(os.path.join(d, "include")):
        # Package is installed
        return os.path.join(d, "include")
    else:
        # Package is from a source directory
        return os.path.join(os.path.dirname(d), "include")
