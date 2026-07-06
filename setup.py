from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Derlenecek C++ kaynak dosyalarının listesi
cpp_sources = [
    "Process.cpp",
    "TrainingLogic.cpp",
    "TestingLogic.cpp", 
    "wrapper.cpp",
]

ext_modules = [
    Pybind11Extension(
        "my_process_module",  # Python'da import edilecek modül adı
        cpp_sources,         
        cxx_std=17,         
    ),
]

setup(
    name="my_process_module",
    version="1.0.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)