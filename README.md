# Adaptive Virtual PIN-Button Recognition on Passive NFC Cards

This repository contains research code and resources for virtual PIN-button recognition on passive NFC cards using the ISO/IEC 15693 protocol. The project combines domain adaptation, machine learning, and conformal prediction to improve recognition accuracy and support adaptive user input across different NFC cards and reader configurations.

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

This project develops a **12-button virtual PIN pad** on unmodified commercial NFC cards. A passive resonant coil is used to select virtual buttons, producing position-dependent RF responses that are recognized using machine learning.

The framework combines **multi-domain adaptation** to handle card and reader variations with **conformal prediction** to determine when an uncertain input requires repetition.

The research builds on **CADRE**, published at IEEE SECON 2026, and extends the framework with uncertainty quantification and adaptive virtual-button recognition.

## Folder Structure

The repository will organize research resources into the following components:

- **Machine Learning:** Deep learning and Random Forest models for virtual-button recognition.
- **Domain Adaptation:** Deep CORAL and CORAL implementations for handling card and reader variations.
- **Conformal Prediction:** Prediction-set calibration and adaptive re-input decisions.
- **Scripts:** Python programs for training, evaluation, and visualization.
- **GNU Radio:** Software-defined NFC communication and RF response collection.
- **Results:** Experimental evaluation and figure-generation scripts.

*Research code and supporting resources will be added as they are prepared and validated.*

## Key Features

- **Virtual PIN Pad:** Twelve virtual buttons on unmodified passive NFC cards.
- **Domain Adaptation:** Robust recognition across different cards and reader configurations.
- **Uncertainty Quantification:** Conformal prediction for uncertainty-aware recognition.
- **Adaptive Re-input:** Additional button inputs are requested when the accumulated evidence is insufficient.
- **ISO/IEC 15693:** NFC communication using a software-defined radio testbed.

## Dependencies and Setup

The implementation uses **Python 3.10** (recommended), machine learning libraries, MAPIE, and GNU Radio.

Specific library versions and installation instructions will be provided with the corresponding research code. The exact Python version used for the reported experiments should be verified before finalizing the reproducibility requirements.

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
