
try:
    import cProfile as profile
except ImportError:
    import profile

import lisptest

profiler = profile.Profile()

exception = False
try:
    try:
        profiler.runcall(lisptest.main)
    except SystemExit: raise
    except:
        exception = True
        raise
finally:
    f = 'raw_stats.profile'
    profiler.dump_stats(f)
#    if not exception:
#        profiler.print_stats()
    import pstats
    s = pstats.Stats(f).strip_dirs().sort_stats('cumu')
    s.print_stats()
    #s.print_callers()

