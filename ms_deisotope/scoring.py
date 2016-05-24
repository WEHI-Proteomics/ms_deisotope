import numpy as np
import operator

from .utils import Base

eps = 1e-4


class IsotopicFitRecord(object):

    __slots__ = ["seed_peak", "score", "charge", "experimental", "theoretical",
                 "monoisotopic_peak", "data", "missed_peaks"]

    def __init__(self, seed_peak, score, charge, theoretical, experimental, data=None,
                 missed_peaks=0, **kwargs):
        self.seed_peak = seed_peak
        self.score = score
        self.charge = charge
        self.experimental = experimental
        self.theoretical = theoretical
        self.monoisotopic_peak = experimental[0]
        self.data = data
        self.missed_peaks = missed_peaks

    def clone(self):
        return self.__class__(
            self.seed_peak, self.score, self.charge, self.theoretical, self.experimental, self.data, self.missed_peaks)

    def __reduce__(self):
        return self.__class__, (
            self.seed_peak, self.score, self.charge, self.theoretical, self.experimental, self.data, self.missed_peaks)

    def __eq__(self, other):
        val = (self.score == other.score and
               self.charge == other.charge and
               self.experimental == other.experimental and
               self.theoretical == other.theoretical)
        if self.data is not None or other.data is not None:
            val = val and (self.data == other.data)
        return val

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    def __hash__(self):
        return hash((self.monoisotopic_peak.mz, self.charge))

    def __iter__(self):
        yield self.score
        yield self.charge
        yield self.experimental
        yield self.theoretical

    @property
    def npeaks(self):
        return len(self.experimental)

    def __repr__(self):
        return "IsotopicFitRecord(score=%0.5f, charge=%d, npeaks=%d, monoisotopic_mz=%0.5f)" % (
            self.score, self.charge, self.npeaks, self.monoisotopic_peak.mz)


class FitSelectorBase(Base):
    minimum_score = 0

    def __init__(self, minimum_score=0):
        self.minimum_score = minimum_score

    def best(self, results):
        return NotImplemented

    def __call__(self, *args, **kwargs):
        return self.best(*args, **kwargs)

    def reject(self, result):
        return NotImplemented

    def is_maximizing(self):
        return False


class MinimizeFitSelector(FitSelectorBase):
    def best(self, results):
        return min(results, key=operator.attrgetter("score"))

    def reject(self, fit):
        return fit.score > self.minimum_score

    def is_maximizing(self):
        return False


class MaximizeFitSelector(FitSelectorBase):
    def best(self, results):
        return max(results, key=operator.attrgetter("score"))

    def reject(self, fit):
        return fit.score < self.minimum_score

    def is_maximizing(self):
        return False


class IsotopicFitterBase(Base):

    def __init__(self, score_threshold=0.5):
        self.select = MinimizeFitSelector(score_threshold)

    def evaluate(self, peaklist, observed, expected, **kwargs):
        return NotImplemented

    def _evaluate(self, peaklist, observed, expected, **kwargs):
        return self.evaluate(peaklist, observed, expected, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def reject(self, fit):
        return self.select.reject(fit)

    def is_maximizing(self):
        return self.select.is_maximizing()


class GTestFitter(IsotopicFitterBase):
    def evaluate(self, peaklist, observed, expected, **kwargs):
        g_score = 2 * sum([obs.intensity * np.log(
            obs.intensity / theo.intensity) for obs, theo in zip(observed, expected)])
        return g_score


g_test = GTestFitter()


class ScaledGTestFitter(IsotopicFitterBase):
    def evaluate(self, peaklist, observed, expected, **kwargs):
        total_observed = sum(p.intensity for p in observed)
        total_expected = sum(p.intensity for p in expected)
        total_expected += eps
        normalized_observed = [obs.intensity / total_observed for obs in observed]
        normalized_expected = [theo.intensity / total_expected for theo in expected]
        g_score = 2 * sum([obs * np.log(obs / theo) for obs, theo in zip(
            normalized_observed, normalized_expected)])
        return g_score


g_test_scaled = ScaledGTestFitter()


class ChiSquareFitter(IsotopicFitterBase):
    def evaluate(self, peaklist, observed, expected, **kwargs):
        score = sum([(obs.intensity - theo.intensity)**2 / theo.intensity
                     for obs, theo in zip(observed, expected)])
        return score


chi_sqr_test = ChiSquareFitter()


class LeastSquaresFitter(IsotopicFitterBase):
    def evaluate(self, peaklist, observed, expected, **kwargs):
        exp_max = max(p.intensity for p in observed)
        theo_max = max(p.intensity for p in expected)

        sum_of_squared_errors = 0
        sum_of_squared_theoreticals = 0

        for e, t in zip(observed, expected):
            normed_expr = e.intensity / exp_max
            normed_theo = t.intensity / theo_max
            sum_of_squared_errors += (normed_theo - normed_expr) ** 2
            sum_of_squared_theoreticals += normed_theo ** 2
        return sum_of_squared_errors / sum_of_squared_theoreticals


least_squares = LeastSquaresFitter()


class MSDeconVFitter(IsotopicFitterBase):
    def __init__(self, minimum_score=10):
        self.select = MaximizeFitSelector()
        self.select.minimum_score = minimum_score

    def calculate_minimum_signal_to_noise(self, observed):
        snr = 0
        n = 0
        for obs in observed:
            if obs.signal_to_noise < 1:
                continue
            snr += obs.signal_to_noise
            n += 1
        return (snr / n) * 0.05

    def reweight(self, obs, theo, obs_total, theo_total):
        norm_obs = obs.intensity / obs_total
        norm_theo = theo.intensity / theo_total
        return norm_obs * np.log(norm_obs / norm_theo)

    def score_peak(self, obs, theo, mass_error_tolerance=0.02, minimum_signal_to_noise=1):
        if obs.signal_to_noise < minimum_signal_to_noise:
            return 0.

        mass_error = np.abs(obs.mz - theo.mz)

        if mass_error <= mass_error_tolerance:
            mass_accuracy = 1 - mass_error / mass_error_tolerance
        else:
            mass_accuracy = 0

        if obs.intensity < theo.intensity and (((theo.intensity - obs.intensity) / obs.intensity) <= 1):
            abundance_diff = 1 - ((theo.intensity - obs.intensity) / obs.intensity)
        elif obs.intensity >= theo.intensity and (((obs.intensity - theo.intensity) / obs.intensity) <= 1):
            abundance_diff = np.sqrt(1 - ((obs.intensity - theo.intensity) / obs.intensity))
        else:
            abundance_diff = 0.
        score = np.sqrt(theo.intensity) * mass_accuracy * abundance_diff
        return score

    def evaluate(self, peaklist, observed, expected, mass_error_tolerance=0.02, **kwargs):
        score = 0
        for obs, theo in zip(observed, expected):
            inc = self.score_peak(obs, theo, mass_error_tolerance, 1)
            score += inc
        return score


class PenalizedMSDeconVFitter(IsotopicFitterBase):
    def __init__(self, minimum_score=10, penalty_factor=1.):
        self.select = MaximizeFitSelector(minimum_score)
        self.msdeconv = MSDeconVFitter()
        self.penalizer = ScaledGTestFitter()
        self.penalty_factor = penalty_factor

    def evaluate(self, peaklist, observed, expected, mass_error_tolerance=0.02, **kwargs):
        score = self.msdeconv.evaluate(observed, expected, mass_error_tolerance)
        penalty = self.penalizer.evaluate(observed, expected)
        return score * (1 - penalty * self.penalty_factor)


def decon2ls_chisqr_test(peaklist, observed, expected, **kwargs):
    fit_total = 0
    sum_total = 0
    for obs, theo in zip(observed, expected):
        intensity_diff = obs.intensity - theo.intensity
        fit_total += (intensity_diff ** 2) / (theo.intensity + obs.intensity)
        sum_total += theo.intensity * obs.intensity
    return fit_total / (sum_total + 0.01)


class InterferenceDetection(object):
    def __init__(self, peaklist):
        self.peaklist = peaklist

    def detect_interference(self, experimental_peaks):
        min_peak = experimental_peaks[0]
        max_peak = experimental_peaks[-1]

        region = self.peaklist.between(
            min_peak.mz - min_peak.full_width_at_half_max,
            max_peak.mz + max_peak.full_width_at_half_max)

        included_intensity = sum(p.intensity for p in experimental_peaks)
        region_intensity = sum(p.intensity for p in region)

        score = 1 - (included_intensity / region_intensity)
        return score


class DistinctPatternFitter(IsotopicFitterBase):

    def __init__(self, minimum_score=0.3):
        self.select = MinimizeFitSelector(minimum_score)
        self.interference_detector = None
        self.g_test_scaled = ScaledGTestFitter()

    def evaluate(self, peaklist, experimental, theoretical):
        npeaks = float(len(experimental))
        if self.interference_detector is None:
            self.interference_detector = InterferenceDetection(peaklist)

        score = self.g_test_scaled(peaklist, experimental, theoretical)
        score *= abs((self.interference_detector.detect_interference(experimental) + 0.01) / (npeaks * 2)) * 100
        return score


try:
    _c = True
    _IsotopicFitRecord = IsotopicFitRecord
    _LeastSquaresFitter = LeastSquaresFitter
    _MSDeconVFitter = MSDeconVFitter
    _ScaledGTestFitter = ScaledGTestFitter
    _PenalizedMSDeconVFitter = PenalizedMSDeconVFitter
    _DistinctPatternFitter = DistinctPatternFitter
    from ._c.scoring import (
        IsotopicFitRecord, LeastSquaresFitter, MSDeconVFitter,
        ScaledGTestFitter, PenalizedMSDeconVFitter, DistinctPatternFitter,
        ScaledPenalizedMSDeconvFitter)
except ImportError, e:
    print e
    _c = False

msdeconv = MSDeconVFitter()
least_squares = LeastSquaresFitter()
g_test_scaled = ScaledGTestFitter()
penalized_msdeconv = PenalizedMSDeconVFitter()
distinct_pattern_fitter = DistinctPatternFitter()
