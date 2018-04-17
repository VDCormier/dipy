import numpy as np
import numpy.testing as npt

from dipy.direction.dispersion_informed_direction_getter import (
    DispersionInformedPeakDirectionGetter)


def test_DispersionInformedPeakDirectionGetter():
    # Test the DispersionInformedPeakDirectionGetter

    # Sample data
    data = np.ones((3, 3, 3, 15), dtype=np.float64)

    # Test that a direction is found
    point = np.zeros((3), dtype=np.float64)
    dir = np.array((0, 0, 1), dtype=np.float64)
    dg = DispersionInformedPeakDirectionGetter(data, 90, 0.1)
    #print dg.cos_similarity
    state = dg.get_direction(point, dir)
    npt.assert_equal(state, 0)

    # Test that no direction is found
    point = np.zeros((3), dtype=np.float64)
    dir = np.array((0, 0, 1), dtype=np.float64)
    dg = DispersionInformedPeakDirectionGetter(data, 1, 0.1)
    state = dg.get_direction(point, dir)
    print dir
    npt.assert_equal(state, 1)

    # Test invalid point
    point = np.array([-2, 0, 0], dtype=np.float64)
    dir = np.array((0, 0, 1), dtype=np.float64)
    dg = DispersionInformedPeakDirectionGetter(data, 1, 0.1)
    state = dg.get_direction(point, dir)
    npt.assert_equal(state, 1)
