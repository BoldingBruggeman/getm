import subprocess
import shutil
from setuptools import Extension
from setuptools.command.build_ext import build_ext
import shlex
import os
from setuptools import setup, find_packages


class CMakeExtension(Extension):
    def __init__(self, name: str, source_dir: str, *cmake_args: str):
        super().__init__(name, sources=[])
        self.source_dir = source_dir
        self.cmake_args = cmake_args


class CMakeBuild(build_ext):
    user_options = build_ext.user_options + [
        ("cmake-opts=", None, "additional options to pass to cmake"),
    ]

    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def initialize_options(self):
        super().initialize_options()
        self.cmake_opts = None

    def build_extension(self, ext: CMakeExtension):
        if not os.path.isdir(self.build_temp):
            os.makedirs(self.build_temp)

        ext_path = self.get_ext_fullpath(ext.name)

        # Directory where your build output should go
        install_prefix = os.path.abspath(os.path.dirname(ext_path))

        # Temporary directory where all intermediate build files should go.
        build_dir = os.path.join(self.build_temp, ext.name)
        if self.force and os.path.isdir(build_dir):
            print(f"Emptying existing build directory {build_dir}")
            shutil.rmtree(build_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)

        build_type = "Debug" if self.debug else "Release"
        cmake_args = list(ext.cmake_args) + [f"-DCMAKE_BUILD_TYPE={build_type}"]
        if self.cmake_opts is not None:
            cmake_args += shlex.split(self.cmake_opts)
        if self.compiler is not None:
            cmake_args.append(f"-DCMAKE_Fortran_COMPILER={self.compiler}")
        build_args = ["--config", build_type]
        cmake_args += [f"-DCMAKE_INSTALL_PREFIX={install_prefix}"]
        build_args += ["--target", "install"]
        subprocess.check_call(
            ["cmake", "-B", build_dir, "-S", ext.source_dir] + cmake_args,
        )
        subprocess.check_call(["cmake", "--build", build_dir] + build_args)


setup(
    packages=find_packages(include=["pyairsea*"]),
    package_data={"pyairsea": ["*.so", "*.dll", "*.dylib", "*.pyd"]},
    ext_modules=[CMakeExtension("pyairsea._pyairsea", os.path.dirname(__file__))],
    zip_safe=False,
    cmdclass={"build_ext": CMakeBuild},
)
