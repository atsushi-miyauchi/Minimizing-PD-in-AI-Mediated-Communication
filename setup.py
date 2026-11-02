"""Build configuration for the optional-by-installation C++ numerical core."""

from __future__ import annotations

import sys

from setuptools import Extension, setup


compile_args = ["/O2", "/std:c++17"] if sys.platform == "win32" else ["-O3", "-std=c++17"]

setup(
    package_dir={"": "src"},
    ext_modules=[
        Extension(
            "polar_ai._cpp_core",
            sources=["cpp/polar_ai_core.cpp"],
            language="c++",
            extra_compile_args=compile_args,
        )
    ]
)
