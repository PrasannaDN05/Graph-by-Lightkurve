# Graph-by-Lightkurve
# 🌌 TESS Exoplanet Transit Detection

A Python-based project that detects potential exoplanet transits using public **TESS** light curve data. The script applies the **Box Least Squares (BLS)** algorithm to identify periodic transit signals and performs basic validation to classify targets as **Transit Candidates** or **No Transit**.

## 🚀 Features

* Downloads TESS light curve data
* Cleans and normalizes observations
* Detects transits using the BLS algorithm
* Calculates Period, Transit Depth, SNR, SDE, and Rp/R*
* Performs Odd-Even and Secondary Eclipse checks
* Generates a phase-folded light curve and BLS periodogram

## 🛠️ Tech Stack

* Python
* Lightkurve
* NumPy
* Matplotlib


## 📌 Future Improvements

* Batch processing of multiple TIC IDs
* CSV export of results
* Machine Learning-based transit classification

---

**Author:** Prasanna D. Nathile
