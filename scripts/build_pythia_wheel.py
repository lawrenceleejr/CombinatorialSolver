"""Build the pythia8mc wheel with a parallel compiler.

`pip install pythia8mc` compiles ~70 large C++ files one at a time (30-60 min).
This runs the same setup.py with the compile step spread over N threads.

  python scripts/build_pythia_wheel.py --jobs 16 --workdir /tmp/pybuild
  pip install /tmp/pybuild/pythia8mc-*/dist/*.whl
"""
import argparse
import glob
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=os.cpu_count())
ap.add_argument("--workdir", default="pybuild")
ap.add_argument("--version", default="8.318.0")
a = ap.parse_args()
os.makedirs(a.workdir, exist_ok=True)
subprocess.check_call([sys.executable, "-m", "pip", "download", f"pythia8mc=={a.version}",
                       "--no-deps", "--no-binary", ":all:", "-d", a.workdir])
sdist = glob.glob(os.path.join(a.workdir, f"pythia8mc-{a.version}.tar.gz"))[0]
with tarfile.open(sdist) as t:
    try:
        t.extractall(a.workdir, filter="data")
    except TypeError:  # Python < 3.12
        t.extractall(a.workdir)
src = os.path.join(a.workdir, f"pythia8mc-{a.version}")

from setuptools._distutils import ccompiler  # noqa: E402


def parallel_compile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0,
                     extra_preargs=None, extra_postargs=None, depends=None):
    macros, objects, extra_postargs, pp_opts, build = self._setup_compile(
        output_dir, macros, include_dirs, sources, depends, extra_postargs)
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)

    def one(obj):
        if obj in build:
            s, ext = build[obj]
            self._compile(obj, s, ext, cc_args, extra_postargs, pp_opts)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(one, objects))
    return objects


ccompiler.CCompiler.compile = parallel_compile
os.chdir(src)
sys.argv = ["setup.py", "bdist_wheel", "-d", "dist"]
exec(compile(open("setup.py").read(), "setup.py", "exec"))
print("wheel:", glob.glob(os.path.join(src, "dist", "*.whl")))
