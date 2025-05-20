import pathlib
import sys

script_dir = pathlib.Path(__file__).parent.absolute()
parent_dir = script_dir.parents[0]
sys.path.append(str(parent_dir))

from labsea_project import tools
import matplotlib
import numpy as np
matplotlib.use('agg')  # use agg backend to prevent creating plot windows during tests

def test_calc_dyn_h():
    # Test with a simple case if specvol_anom has only one vertical layer output should be zero
    specvol = np.ones((1, 3))
    p = np.ones((1, 3)) * 10
    dyn_h = tools.calc_dyn_h(specvol, p)
    assert np.allclose(dyn_h, 0)