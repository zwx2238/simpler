
import os
import shutil
import subprocess
from enum import IntEnum
from typing import List, Optional
import env_manager


# Must match compile_strategy.h
class ToolchainType(IntEnum):
    """Toolchain types matching the C enum in compile_strategy.h."""
    CCEC = 0           # ccec (Ascend AICore compiler)
    HOST_GXX_15 = 1    # g++-15 (host, simulation kernels)
    HOST_GXX = 2       # g++ (host, orchestration .so)
    AARCH64_GXX = 3    # aarch64-target-linux-gnu-g++ (cross-compile)


class Toolchain:
    """Base class for all compile toolchains.

    A Toolchain represents a compiler identity: which compiler binary to use,
    what flags to pass, and what CMake -D arguments to generate.

    The Ascend SDK path is managed by env_manager. Call
    env_manager.ensure("ASCEND_HOME_PATH") before creating toolchains that
    need the Ascend SDK (CCECToolchain, Aarch64GxxToolchain, GxxToolchain
    with Ascend includes).

    Used by:
    - KernelCompiler: calls get_compile_flags() for direct single-file invocation
    - BuildTarget (in runtime_compiler.py): calls get_cmake_args() for CMake builds
    """

    def __init__(self):
        self.ascend_home_path = env_manager.get("ASCEND_HOME_PATH")

    def get_compile_flags(self, **kwargs) -> List[str]:
        """Return base compiler flags for direct invocation."""
        raise NotImplementedError

    def get_cmake_args(self) -> List[str]:
        """Return compiler-specific CMake -D arguments."""
        raise NotImplementedError


def _resolve_tool_path(
    override_env: str,
    candidate_names: List[str],
    candidate_paths: List[str],
) -> Optional[str]:
    override = os.environ.get(override_env)
    if override and os.path.isfile(override):
        return override

    for name in candidate_names:
        resolved = shutil.which(name)
        if resolved:
            return resolved

    for path in candidate_paths:
        if path and os.path.isfile(path):
            return path

    return override


class CCECToolchain(Toolchain):
    """Ascend ccec compiler for AICore kernels."""

    def __init__(self, platform: str = "a2a3"):
        super().__init__()
        self.platform = platform

        self.cxx_path = _resolve_tool_path(
            "SETUP_ASCEND_BISHENG_BIN",
            ["bisheng"],
            [
                os.path.join(self.ascend_home_path, "bin", "bisheng"),
                os.path.join(self.ascend_home_path, "compiler", "ccec_compiler", "bin", "bisheng"),
            ],
        )
        self.linker_path = _resolve_tool_path(
            "SETUP_ASCEND_LD_LLD_BIN",
            ["ld.lld"],
            [
                os.path.join(self.ascend_home_path, "bin", "ld.lld"),
                os.path.join(self.ascend_home_path, "compiler", "ccec_compiler", "bin", "ld.lld"),
            ],
        )

        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(
                f"bisheng compiler not found: {self.cxx_path}"
            )
        if not os.path.isfile(self.linker_path):
            raise FileNotFoundError(
                f"ccec linker not found: {self.linker_path}"
            )

    def get_compile_flags(self, core_type: str = "aiv", **kwargs) -> List[str]:
        # A5 uses dav-c310 architecture, A2A3 uses dav-c220
        if self.platform in ("a5", "a5sim"):
            arch = "dav-c310-vec" if core_type == "aiv" else "dav-c310-cube"
        elif self.platform in ("a2a3", "a2a3sim"):
            arch = "dav-c220-vec" if core_type == "aiv" else "dav-c220-cube"
        else:
            raise ValueError(f"Unknown platform: {self.platform}. Supported: a2a3, a2a3sim, a5, a5sim")

        return [
            "-c", "-O3", "-g", "-x", "cce",
            "-Wall", "-std=c++17",
            "--cce-aicore-only",
            f"--cce-aicore-arch={arch}",
            "-mllvm", "-cce-aicore-stack-size=0x8000",
            "-mllvm", "-cce-aicore-function-stack-size=0x8000",
            "-mllvm", "-cce-aicore-record-overflow=false",
            "-mllvm", "-cce-aicore-addr-transform",
            "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
            "-DMEMORY_BASE",
        ]

    def get_cmake_args(self) -> List[str]:
        return [
            f"-DBISHENG_CC={self.cxx_path}",
            f"-DBISHENG_LD={self.linker_path}",
        ]


class Gxx15Toolchain(Toolchain):
    """g++-15 compiler for simulation kernels."""

    def __init__(self):
        super().__init__()
        configured_cxx = os.environ.get("CXX")
        self.cxx_path = (
            shutil.which(configured_cxx) if configured_cxx else None
        ) or shutil.which("g++-15") or shutil.which("g++") or configured_cxx or "g++-15"
        self.cxx_std_flag = self._choose_cpp_std_flag()

    def _choose_cpp_std_flag(self) -> str:
        if self.cxx_path == "g++-15":
            return "-std=c++23"

        try:
            result = subprocess.run(
                [self.cxx_path, "-std=c++23", "-x", "c++", "-E", "-"],
                input="",
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return "-std=c++23"
        except OSError:
            pass

        return "-std=gnu++2a"

    def get_compile_flags(self, core_type: str = "", **kwargs) -> List[str]:
        flags = [
            "-shared", "-O2", "-fPIC",
            self.cxx_std_flag,
            "-fpermissive",
            "-Wno-macro-redefined",
            "-Wno-ignored-attributes",
            "-D__CPU_SIM",
            "-DPTO_CPU_MAX_THREADS=1",
            "-DNDEBUG",
        ]
        # g++ does not define __DAV_VEC__/__DAV_CUBE__ like ccec does,
        # so we must add them explicitly based on core_type.
        if core_type == "aiv":
            flags.append("-D__DAV_VEC__")
        elif core_type == "aic":
            flags.append("-D__DAV_CUBE__")
        return flags

    def get_cmake_args(self) -> List[str]:
        # Respect CC/CXX environment variables (e.g., CXX=g++-15 on macOS CI)
        cc = os.environ.get("CC", "gcc")
        cxx = os.environ.get("CXX", "g++")
        args = [
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]
        return args


class GxxToolchain(Toolchain):
    """g++ compiler for host compilation."""

    def __init__(self):
        super().__init__()
        configured_cxx = os.environ.get("CXX")
        self.cxx_path = (shutil.which(configured_cxx) if configured_cxx else None) or "g++"

    def get_compile_flags(self, **kwargs) -> List[str]:
        return ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]

    def get_cmake_args(self) -> List[str]:
        # Respect CC/CXX environment variables (e.g., CXX=g++-15 on macOS CI)
        cc = os.environ.get("CC", "gcc")
        cxx = os.environ.get("CXX", "g++")
        args = [
            f"-DCMAKE_C_COMPILER={cc}",
            f"-DCMAKE_CXX_COMPILER={cxx}",
        ]
        if self.ascend_home_path:
            args.append(f"-DASCEND_HOME_PATH={self.ascend_home_path}")
        return args


class Aarch64GxxToolchain(Toolchain):
    """aarch64 cross-compiler for device code."""

    def __init__(self):
        super().__init__()
        self.cxx_path = _resolve_tool_path(
            "SETUP_ASCEND_HCC_GXX_BIN",
            ["aarch64-target-linux-gnu-g++"],
            [
                os.path.join(
                    self.ascend_home_path, "toolkit", "toolchain", "hcc", "bin",
                    "aarch64-target-linux-gnu-g++",
                ),
            ],
        )
        self.cc_path = _resolve_tool_path(
            "SETUP_ASCEND_HCC_GCC_BIN",
            ["aarch64-target-linux-gnu-gcc"],
            [
                os.path.join(
                    self.ascend_home_path, "toolkit", "toolchain", "hcc", "bin",
                    "aarch64-target-linux-gnu-gcc",
                ),
            ],
        )
        if not os.path.isfile(self.cc_path):
            raise FileNotFoundError(
                f"aarch64 C compiler not found: {self.cc_path}"
            )
        if not os.path.isfile(self.cxx_path):
            raise FileNotFoundError(
                f"aarch64 C++ compiler not found: {self.cxx_path}"
            )

    def get_compile_flags(self, **kwargs) -> List[str]:
        return ["-shared", "-fPIC", "-O3", "-g", "-std=c++17"]

    def get_cmake_args(self) -> List[str]:
        return [
            f"-DCMAKE_C_COMPILER={self.cc_path}",
            f"-DCMAKE_CXX_COMPILER={self.cxx_path}",
            f"-DASCEND_HOME_PATH={self.ascend_home_path}",
        ]
