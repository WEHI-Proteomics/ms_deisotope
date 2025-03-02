.. image:: https://raw.githubusercontent.com/mobiusklein/ms_deisotope/master/docs/_static/logo.png

A Library for Deisotoping and Charge State Deconvolution For Mass Spectrometry
------------------------------------------------------------------------------

This library combines `brainpy` and `ms_peak_picker` to build a toolkit for
MS and MS/MS data. The goal of these libraries is to provide pieces of the puzzle
for evaluating MS data modularly. The goal of this library is to combine the modules
to streamline processing raw data.


Installing
----------

Building from source requires a version of Cython >= 0.27.0


API
---


Data Access
===========

``ms_deisotope`` can read from mzML, mzXML and MGF files directly, using the ``pyteomics`` library.
On Windows, it can also use ``comtypes`` to access Thermo's MSFileReader.dll to read RAW files and
Agilent's MassSpecDataReader.dll to read .d directories. Whenever possible, the library provides a
common interface to all supported formats. With Thermo's pure .NET library, it can use ``pythonnet``
to read Thermo RAW files on Windows and Linux (and presumably Mac, too).

.. code:: python

    from ms_deisotope import MSFileReader
    from ms_deisotope.data_source import mzxml

    # open a file, selecting the appropriate reader automatically
    reader = MSFileReader("path/to/data.mzML")

    # or specify the reader type directly
    reader = mzxml.MzXMLLoader("path/to/data.mzXML")


All supported readers provide fast random access for uncompressed files, and support the Iterator
interface.

.. code:: python

    # jump the iterator to the MS1 scan nearest to 30 minutes into the run
    reader.start_from_scan(rt=30)

    # read out the next MS1 scans and all associated MSn scans
    scan_bunch = next(reader)
    print(scan_bunch.precursor, len(scan_bunch.products))


Averagine
=========

An "Averagine" model is used to describe the composition of an "average amino acid",
which can then be used to approximate the composition and isotopic abundance of a
combination of specific amino acids. Given that often the only solution available is
to guess at the composition of a particular *m/z* because there are too many possible
elemental compositions, this is the only tractable solution.

This library supports arbitrary Averagine formulae, but the Senko Averagine is provided
by default: `{"C": 4.9384, "H": 7.7583, "N": 1.3577, "O": 1.4773, "S": 0.0417}`

.. code:: python

    from ms_deisotope import Averagine
    from ms_deisotope import plot

    peptide_averagine = Averagine({"C": 4.9384, "H": 7.7583, "N": 1.3577, "O": 1.4773, "S": 0.0417})

    plot.draw_peaklist(peptide_averagine.isotopic_cluster(1266.321, charge=1))


`ms_deisotope` includes several pre-defined averagines (or "averagoses" as may be more appropriate):
    1. Senko's peptide - `ms_deisotope.peptide`
    2. Native *N*- and *O*-glycan - `ms_deisotope.glycan`
    3. Permethylated glycan - `ms_deisotope.permethylated_glycan`
    4. Glycopeptide - `ms_deisotope.glycopeptide`
    5. Sulfated Glycosaminoglycan - `ms_deisotope.heparan_sulfate`
    6. Unsulfated Glycosaminoglycan - `ms_deisotope.heparin`

Deconvolution
=============

The general-purpose averagine-based deconvolution procedure can be called by using the high level
API function `deconvolute_peaks`, which takes a sequence of peaks, an averagine model, and a isotopic
goodness-of-fit scorer:

.. code:: python

    import ms_deisotope

    deconvoluted_peaks, _ = ms_deisotope.deconvolute_peaks(peaks, averagine=ms_deisotope.peptide,
                                                           scorer=ms_deisotope.MSDeconVFitter(10.))

The result is a deisotoped and charge state deconvoluted peak list where each peak's neutral mass is known
and the fitted charge state is recorded along with the isotopic peaks that gave rise to the fit.

Refer to the documentation for a deeper description of isotopic pattern fitting.