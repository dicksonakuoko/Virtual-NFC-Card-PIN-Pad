# Adaptive Virtual PIN-Button Recognition on Passive NFC Cards

This repository contains research code and resources for virtual PIN-button recognition on passive NFC cards using the ISO/IEC 15693 protocol. The project combines machine learning, domain adaptation, and conformal prediction to improve recognition reliability across different NFC cards and reader configurations.

---

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Key Features](#key-features)
- [Dependencies and Setup](#dependencies-and-setup)
- [Publications](#publications)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project develops a **12-button virtual PIN pad** on unmodified commercial NFC cards. A passive resonant coil is placed over designated button regions, producing position-dependent RF responses that are recognized using machine learning.

The framework uses **multi-domain adaptation** to improve recognition across different NFC cards and readers. It further incorporates **conformal prediction** to quantify recognition uncertainty and request additional input when the available evidence is insufficient.

This research builds on **CADRE**, published at IEEE SECON 2026, by extending the domain adaptation framework with uncertainty quantification and adaptive virtual-button recognition.

## Folder Structure

The repository is organized into three main folders:

### Files

Contains supporting research files and resources related to the project.

### GNU Radio

Contains GNU Radio programs for NFC communication and RF response collection.

### Script

Contains Jupyter notebooks for dataset preparation, deep learning and Random Forest training, domain adaptation, conformal prediction, and adaptive virtual-button recognition.

## Key Features

- **Virtual PIN Pad:** Enables twelve virtual buttons on unmodified passive NFC cards.
- **Domain Adaptation:** Improves recognition across different NFC cards and reader configurations.
- **Uncertainty Quantification:** Uses conformal prediction to evaluate recognition uncertainty.
- **Adaptive Re-input:** Requests additional input when the accumulated evidence is insufficient.
- **ISO/IEC 15693:** Uses software-defined radio for standard-compliant NFC communication.

## Dependencies and Setup

The implementation uses **Python 3.10** (recommended), TensorFlow, scikit-learn, MAPIE, NumPy, and other scientific Python libraries.

The Jupyter notebooks use NFC response datasets collected through GNU Radio. The corresponding datasets and pretrained models must be available locally to run the experiments.

Specific library versions and installation requirements depend on the experimental environment.

## Publications

### Conference Paper — CADRE

**D. A. Sarpong and H. Guo**, "CADRE: Card-Agnostic Domain-Aligned RF Embeddings for Virtual PIN Pads on Passive NFC Cards," in *2026 22nd Annual IEEE International Conference on Sensing, Communication, and Networking (SECON)*, pp. 332–340, 2026.

**DOI:** [10.1109/SECON68281.2026.11579229](https://doi.org/10.1109/SECON68281.2026.11579229)

## Citation

If you use this repository or its research methods in your work, please cite the published CADRE paper:

```bibtex
@inproceedings{Sarpong2026CADRE,
  author    = {Dickson Akuoko Sarpong and Hongzhi Guo},
  title     = {{CADRE}: Card-Agnostic Domain-Aligned
               {RF} Embeddings for Virtual {PIN} Pads
               on Passive {NFC} Cards},
  booktitle = {2026 22nd Annual IEEE International
               Conference on Sensing, Communication,
               and Networking (SECON)},
  year      = {2026},
  pages     = {332--340},
  doi       = {10.1109/SECON68281.2026.11579229}
}
```

The journal extension citation will be added once publication details are available.

## Contact

For questions about the project:

**Dickson Akuoko Sarpong**

School of Computing, University of Nebraska–Lincoln

Email: dakuokosarpong2@huskers.unl.edu

## Acknowledgments

**WISE Lab**, School of Computing, University of Nebraska–Lincoln.

This research builds on the CADRE framework developed for virtual PIN pads on passive NFC cards.
