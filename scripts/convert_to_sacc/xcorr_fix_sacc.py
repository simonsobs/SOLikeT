import os

import numpy as np
import sacc

tests_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
s = sacc.Sacc.load_fits(
    os.path.join(tests_path, "tests/data/unwise_g-so_kappa.sim.sacc.fits")
)

tracer: sacc.tracers.NZTracer = s.get_tracer("gc_unwise")

dndz_filename = "soliket/xcorr/data/dndz.txt"
dndz = np.loadtxt(os.path.join(tests_path, dndz_filename))

assert np.allclose(tracer.z, dndz[:, 0])
assert np.allclose(tracer.nz, dndz[:, 1])

tracer.extra_columns = {"dndz": dndz[:, 1]}

s.add_tracer_object(tracer)

s.save_fits(
    os.path.join(tests_path, "tests/data/unwise_g-so_kappa.sim.sacc.fits"),
    overwrite=True,
)
