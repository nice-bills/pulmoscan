# PulmoScan: Medical Image Analysis Portfolio

**Goal:** High-performance COVID-19 detection from X-ray images, demonstrating advanced backend optimization (Caching, Batching, Async) and mobile deployment (ONNX).

## 📅 Project Context
- **Start Date:** Dec 7, 2025
- **OS:** Windows (win32)
- **Environment:** `uv` managed

## 🚀 Active Phase: Phase 1 (Setup & Training)
- [ ] Initialize Environment (User running `uv init`)
- [x] Data Setup (Filtered COVID, Normal, Viral Pneumonia)
- [ ] Install Dependencies (CPU versions for Torch)
- [ ] Train Baseline Model (MobileNetV3)

## 🧠 Memory Bank
- **Dataset:** `data/covid19` (Classes: COVID, Normal, Viral Pneumonia). `Lung_Opacity` excluded.
- **Model:** MobileNetV3-Large.
- **Key Constraints:**
    - User runs all `uv` commands.
    - **PyTorch:** CPU version (optimization/size).
    - Focus is on *System Engineering* (Redis/Celery) over pure ML research.

## 📝 Recent Actions
- Created project `pulmoscan`.
- Imported and filtered dataset from Downloads.
