# EdgeAI-Metabolic-DigitalTwin

[![Hardware: Jetson Orin Nano](https://img.shields.io/badge/Hardware-Nvidia%20Jetson%20Orin%20Nano-76B900?logo=nvidia)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![Hardware: ESP32](https://img.shields.io/badge/Hardware-ESP32-E7352C?logo=espressif)](https://www.espressif.com/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Database: AI-READI v2](https://img.shields.io/badge/Dataset-AI--READI%20v2-blue)](https://aireadi.org/)

A highly-optimized, **3-Layer Hybrid Physiological Digital Twin** designed to predict real-time glycemic variability in Type 2 Diabetes (T2D) patients.

Deployed locally on an **Nvidia Jetson Orin Nano** edge compute node, this platform bridges mathematical biological modeling with deep learning to deliver highly accurate, clinically-explainable blood glucose forecasting.

---

## 🧠 System Architecture

Instead of relying on a "black-box" neural network to predict metabolic patterns from scratch, this project implements a dual-core hybrid topology:

- **Layer 1 (The Biophysical Core):** Integrates a modified UVa/Padova system of ordinary differential equations (ODEs) using the `simglucose` framework to establish a safe, medically-bounded glucose baseline.
- **Layer 2 (The Data-Driven AI Core):** A lightweight **PyTorch LSTM** trained on the multi-modal clinical **AI-READI v2 dataset** to continuously predict the glycemic residual error ($\Delta G$) caused by real-world behavioral friction (e.g., stress, exercise, nutrition).
- **Layer 3 (The Edge Integration Core):** A native **FastAPI** server that ingests real-time behavioral streams from an **ESP32** wearable simulator and processes meals using a custom, localized South Asian Dietary Ontology.

---

## 🛠️ Tech Stack & Key Specs

- **Host Platform:** Nvidia Jetson Orin Nano Developer Kit (ARM Cortex-A78AE CPU, Ampere GPU, Tensor Cores)
- **Wearable Simulator:** ESP32 (MicroPython/C++ streaming real-time physical metrics)
- **Deep Learning:** PyTorch (Trained locally on Windows/Azure, optimized for Jetson edge deployment)
- **Physical Modeling:** `simglucose` (UVa/Padova simulator adapted with customized insulin sensitivity coefficients for T2D)
- **Deployment Pipeline:** FastAPI, Git version-controlled, localized regional food ontology databases
