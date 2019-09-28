
#!/usr/local/bin/python
"""
build and run a .pyx file with dependencies.
runs a main() function if it exists.
automatically detects 'cimport numpy'
and includes headers. other includes/libs specificed by command-line args.
    %prog [options] source.pyx dep1.c dep2.c
example (note library inclusion is via -l):
    %prog --time --cpp -lstdc++ rectangle.pyx
and %prog will find any .cpp files in the directory and include them.
or dependencies can be specified explicitly:
    %prog --time --cpp -lstdc++ rectangle.pyx Rectangle.cpp
the --time will time the import and execution of the module including a main()
function if it exists.
simplest use with a 't.pyx' file containing:
    def main():
        cdef double sum = 0
        cdef long i
        for i in range(599499400):
            sum += 1.0
        print sum
is:
    %prog cythonrun --time t.pyx
"""
import optparse
import os.path as op
import sys
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import Cython.Compiler.Options
import glob
import os
import time
Cython.Compiler.Options.embed = True

def rm(path):
    if op.exists(path): os.unlink(path)

def get_includes(pyx, includes):
    dirname = op.dirname(pyx)
    include_dirs = [".", dirname]

    if 'cimport numpy' in open(pyx).read():
        import numpy as np
        include_dirs.append(np.get_include())
    include_dirs.extend(includes)

    return include_dirs

def get_sources(pyx, cpp):
    ext = ("*.cpp" if cpp else "*.c")
    sources = [pyx]
    dirname = op.dirname(pyx) + "/"
    sources.extend(glob.glob(dirname + ext))
    return [s for s in sources if s != op.splitext(pyx)[0] + ext[1:]]


def main(source_files, cpp, dotime, **kwargs):
    pyx = op.abspath(source_files[0])
    assert op.exists(pyx), "%s does not exist" % pyx

    # if args[1:] were specified, use those. other wise check for
    # *.c/cpp in same dir as source file.
    sources = get_other_files(source_files, cpp) \
            if len(source_files) > 1 else get_sources(pyx, cpp)

    dirname = op.dirname(pyx)
    cfile = op.splitext(pyx)[0] + (".cpp" if cpp else ".c")
    rm(cfile); rm(op.splitext(pyx)[0] + ".so")

    if not kwargs.get('verbose'):
        _err, _out = sys.stderr, sys.stdout
        sys.stdout = sys.stderr = open('cythonrun.build.log', 'w')

    include_dirs = get_includes(pyx, kwargs['includes'])
    #sources = get_sources(pyx, cpp) + other_files

    module = op.splitext(op.basename(pyx))[0]
    sys.argv = [sys.argv[0], "build_ext", "-i"]
    ext = Extension(module,
                      sources=sources,
                      include_dirs=include_dirs,
                      language="c++" if cpp else 'c',
                      libraries=kwargs.get('libraries', []))

    #ext.line_pyrex_directives = {'embed': None}
    try:
        setup(
            name=module,
            ext_modules=[ext],
            cmdclass = {'build_ext': build_ext},
        )
        if not kwargs.get('verbose'):
            sys.stderr, sys.stdout = _err, _out
        run_module(dirname, module, dotime)
    finally:
        pass
        #rm(cfile); rm(module + ".so")

def run_module(dirname, module, dotime):
    sys.path.insert(0, dirname)

    t = time.time()
    mod = __import__(module)
    try:
        mod.main()
    except AttributeError:
        pass
    if dotime:
        print >>sys.stderr, "execution time: %.3f" % (time.time() - t)

def get_other_files(source_files, cpp):
    """
    extra_files is a list of files or directories or  glob patterns
    """
    pyx = source_files[0]
    ext = '.cpp' if cpp else '.c'
    ofs = [pyx]
    for f in source_files[1:]:
        if "*" in f: ofs.extend(map(op.abspath, glob.glob(f)))
        elif op.isdir(f):
            ofs.extend(map(op.abspath, glob.glob(f + "/*" + ext)))
        elif op.exists(f):
            ofs.append(op.abspath(f))
    return ofs

if __name__ == "__main__":
    p = optparse.OptionParser(__doc__)
    p.add_option("-l", dest="libs", help="libraries", default="")
    p.add_option("-i", "-I", dest="includes", help="include dirs", default="")
    p.add_option("--cpp", "--cplus", dest="cpp", help="use cpp",
                 action="store_true", default=False)
    p.add_option("-t", "--time", dest="time", help="time execution of main"
                 " in compiled module", action='store_true', default=False)
    p.add_option("-v", "--verbose", dest="verbose", help="show output from building module"
                 , action='store_true', default=False)

    kwargs = {}

    opts, source_files = p.parse_args()
    if not len(source_files) >= 1:
        sys.exit(p.print_help())
    kwargs['verbose'] = opts.verbose
    kwargs['libraries'] = [x for x in opts.libs.split(",") if x.strip()]
    kwargs['includes'] = [x for x in opts.includes.split(",") if x.strip()]
    main(source_files, opts.cpp, opts.time, **kwargs)