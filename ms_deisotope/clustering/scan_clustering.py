import functools
import bisect

from .peak_set_similarity import peak_set_similarity


def ppm_error(x, y):
    return (x - y) / y


@functools.total_ordering
class SpectrumCluster(object):
    def __init__(self, scans=None):
        if scans is None:
            scans = []
        self.scans = scans

    @property
    def neutral_mass(self):
        return self.scans[0].precursor_information.neutral_mass

    def __lt__(self, other):
        return self.neutral_mass < other.neutral_mass

    def __gt__(self, other):
        return self.neutral_mass > other.neutral_mass

    def __eq__(self, other):
        return (abs(self.neutral_mass - other.neutral_mass) / other.neutral_mass) < 1e-6

    def __repr__(self):
        return "SpectrumCluster(%f, %d)" % (self.neutral_mass, len(self))

    def __iter__(self):
        return iter(self.scans)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, i):
        return self.scans[i]

    def append(self, item):
        self.scans.append(item)

    def average_similarity(self, *args, **kwargs):
        n = len(self)
        if n == 1:
            return 1.0
        ratings = []
        for i in range(n):
            scan_i = self[i]
            for j in range(i + 1, n):
                scan_j = self[j]
                ratings.append(peak_set_similarity(scan_i, scan_j, *args, **kwargs))
        return sum(ratings) / len(ratings)


def binsearch(array, x):
    n = len(array)
    lo = 0
    hi = n

    while hi != lo:
        mid = (hi + lo) // 2
        y = array[mid].neutral_mass
        err = y - x
        # Do refinement in :func:`cluster_scans`
        if hi - lo == 1:
            return mid
        elif err > 0:
            hi = mid
        else:
            lo = mid
    return 0


def cluster_scans(scans, precursor_error_tolerance=1e-5, minimum_similarity=0.1):
    scans = sorted(scans, key=lambda x: sum(p.intensity for p in x.peak_set), reverse=True)
    # the first scan is always the start of a new cluster
    clusters = [SpectrumCluster([scans[0]])]
    for scan in scans[1:]:
        best_cluster = None
        best_similarity = 0.0

        center_i = binsearch(clusters, scan.precursor_information.neutral_mass)
        i = center_i

        while i >= 0:
            cluster = clusters[i]
            if abs(ppm_error(scan.precursor_information.neutral_mass,
                             cluster.neutral_mass)) > precursor_error_tolerance:
                break
            similarity = peak_set_similarity(scan, cluster[0])
            i -= 1
            if similarity > best_similarity and similarity > minimum_similarity:
                best_similarity = similarity
                best_cluster = cluster

        n = len(clusters)
        i = center_i + 1
        while i < n:
            cluster = clusters[i]
            if abs(ppm_error(scan.precursor_information.neutral_mass,
                             cluster.neutral_mass)) > precursor_error_tolerance:
                break
            similarity = peak_set_similarity(scan, cluster[0])
            i += 1
            if similarity > best_similarity and similarity > minimum_similarity:
                best_similarity = similarity
                best_cluster = cluster

        if best_cluster is None:
            cluster = SpectrumCluster([scan])
            bisect.insort(clusters, cluster)
        else:
            best_cluster.append(scan)
    return clusters


def iterative_clustering(scans, precursor_error_tolerance=1e-5, similarity_thresholds=None):
    if similarity_thresholds is None:
        similarity_thresholds = [0.1, .4, 0.7]
    singletons = []
    to_bisect = [scans]
    for similarity_threshold in similarity_thresholds:
        next_to_bisect = []
        for group in to_bisect:
            clusters = cluster_scans(
                group, precursor_error_tolerance, minimum_similarity=similarity_threshold)
            for cluster in clusters:
                if len(cluster) == 1:
                    singletons.append(cluster)
                else:
                    next_to_bisect.append(cluster)
        to_bisect = next_to_bisect
    return sorted(list(singletons) + list(to_bisect))


class ScanClusterWriter(object):
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)

    def save(self, cluster):
        self.write("%f\t%d\n" % (cluster.neutral_mass, len(cluster)))
        for member in cluster:
            self.write("\t%s\n" % member.id)
        self.write('\n')
