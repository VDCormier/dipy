cimport numpy as np
import numpy as np

from dipy.direction.closest_peak_direction_getter cimport closest_peak
from dipy.tracking.local.direction_getter cimport DirectionGetter
#from dipy.tracking.local.interpolation import trilinear_interpolate4d_c

cdef class DispersionInformedPeakDirectionGetter(DirectionGetter):

    cdef:
        double peak_threshold
        double cos_similarity
        int nbr_peaks
        double[:, :, :, :] data
        np.ndarray peaks

    def __init__(self, data, max_angle, peak_threshold, **kwargs):
        """Create a Dispersion Informed DirectionGetter

        Parameters
        ----------
        data : ndarray, float, (..., N*3)
            Peaks data with N peaks per voxel.
            last dimention format: [Px1,Py1,Pz1, ..., PxN,PyN,PzN]
        max_angle : float (0, 90)
            Maximum angle between tract segments. This angle can be more
            generous (larger) than values typically used with probabilistic
            direction getters.
        peak_threshold : float
            Threshold for peak norm.
        """
        if data.shape[-1] % 3 != 0:
            raise ValueError("data should be a 4d array of N peaks, with the "
                             "last dimension of size N*3")

        self.nbr_peaks = data.shape[-1]/3
        self.peaks = np.zeros((self.nbr_peaks, 3), dtype=float)
        for i in range(self.nbr_peaks):
            norm= np.linalg.norm(data[:,:,:,i*3:(i+1)*3], axis=3)
            for j in range(3):
                data[:,:,:,i*3+j] =data[:,:,:,i*3+j] / norm

        self.data = np.asarray(data,  dtype=float)
        self.peak_threshold = peak_threshold
        self.cos_similarity = np.cos(np.deg2rad(max_angle))
        print "init", self.cos_similarity, max_angle

    cpdef np.ndarray[np.float_t, ndim=2] initial_direction(self,
                                                           double[::1] point):
        """Returns best directions at seed location to start tracking.

        Parameters
        ----------
        point : ndarray, shape (3,)
            The point in an image at which to lookup tracking directions.

        Returns
        -------
        directions : ndarray, shape (N, 3)
            Possible tracking directions from point. ``N`` may be 0, all
            directions should be unique.

        """
        cdef:
            int p[3]

        for i in range(3):
            p[i] = int(point[i] + 0.5)

        for i in range(3):
            if p[i] < -.5 or p[i] >= (self.data.shape[i] - .5):
                return None
        for i in range(self.nbr_peaks):
            self.peaks[i, :] = self.data[p[0], p[1], p[2], i*3:(i+1)*3]

        return self.peaks

    cdef int get_direction_c(self, double* point, double* direction):
        """

        Returns
        -------
        status : int
            Returns 0 `direction` was updated with a new tracking direction, or
            1 otherwise.
        """
        cdef:
            int p[3]
            double d[3]

        for i in range(3):
            p[i] = int(point[i] + 0.5)

        for i in range(3):
            if p[i] < -.5 or p[i] >= (self.data.shape[i] - .5):
                return 1
        for i in range(self.nbr_peaks):
            self.peaks[i, :] = self.data[p[0], p[1], p[2], i*3:(i+1)*3]

        return closest_peak(self.peaks, direction, self.cos_similarity)
