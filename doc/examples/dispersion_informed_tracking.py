"""
================================================================
An introduction to the Dispersion Informed Peak Direction Getter
================================================================

....
"""

from dipy.data import read_stanford_labels, get_sphere
from dipy.reconst.peaks import (peaks_from_model,
                                reshape_peaks_for_visualization)
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.tracking import utils
from dipy.tracking.local import (ThresholdTissueClassifier, LocalTracking)

hardi_img, gtab, labels_img = read_stanford_labels()
data = hardi_img.get_data()
labels = labels_img.get_data()
affine = hardi_img.affine

seed_mask = labels == 2
white_matter = (labels == 1) | (labels == 2)
seeds = utils.seeds_from_mask(seed_mask, density=1, affine=affine)

csd_model = ConstrainedSphericalDeconvModel(gtab, None, sh_order=4)
sphere = get_sphere('repulsion724')
pfm = peaks_from_model(model=csd_model,
                       data=data,
                       sphere=sphere,
                       relative_peak_threshold=.5,
                       min_separation_angle=25,
                       return_sh=False,
                       normalize_peaks=True,
                       parallel=True)

peaks = reshape_peaks_for_visualization(pfm)

"""
We use the fractional anisotropy (FA) of the DTI model to build a tissue
classifier.
"""

import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy

tensor_model = dti.TensorModel(gtab)
tenfit = tensor_model.fit(data, mask=white_matter)

FA = fractional_anisotropy(tenfit.evals)
classifier = ThresholdTissueClassifier(FA, .2)

"""
The Peaks....
"""

from dipy.data import default_sphere
from dipy.direction import DispersionInformedPeakDirectionGetter
from dipy.io.trackvis import save_trk

dg = DispersionInformedPeakDirectionGetter(peaks,
                                           max_angle=60.,
                                           peak_threshold=0.1)
streamlines = LocalTracking(dg, classifier, seeds, affine, step_size=.5,
                            max_cross=1)

save_trk("dispersionInformed.trk", streamlines, affine, labels.shape)
