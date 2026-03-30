# 🔍 Forensic Audio Authentication System

> A production-ready deep learning pipeline for detecting audio tampering, splicing,
> and deepfake injection — engineered for legal and investigative applications.

---

## How It Works
```
Raw Audio → LFCC Features → CNN (spectral patterns) → BiLSTM (temporal context) → Softmax → Verdict
```

---

## Getting Started
```bash
cd forensic_audio_auth

# 2. Set up a virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt


# 5. Generate tampered clips (1 000 by default)
python tampering/generate_dataset.py

# 6. Build train/val/test splits
python features/preprocess.py

# 7. Train the model
python model/train.py

# 8. Evaluate performance
python model/evaluate.py

# 9. Launch the web interface
python -m app.app
```

---

## Datasets

| Dataset | Label | Source |
|---|---|---|
| TIMIT | Authentic (0) | LDC / OpenSLR |
| ASVspoof2019 LA | Tampered (1) | [datashare.ed.ac.uk](https://datashare.ed.ac.uk/handle/10283/3336) |
| WaveFake | Tampered (1) | [github.com/RUB-SysSec/WaveFake](https://github.com/RUB-SysSec/WaveFake) |
| MUSAN (noise augmentation) | — | [openslr.org/17](https://openslr.org/17/) |

---

## Performance Targets

| Metric | Target |
|---|---|
| Equal Error Rate (EER) | < 5% |
| AUC-ROC | > 0.95 |
| Precision @ 1% FAR | > 80% |

---

## Project Structure
```
forensic_audio_auth/
├── data/               Raw audio files, organized by class
├── features/           LFCC extraction pipeline + PyTorch Dataset
├── tampering/          Automated tampering simulation scripts
├── model/              CNN-BiLSTM architecture, training loop, evaluation
├── xai/                Explainability — Grad-CAM & SHAP
├── app/                Gradio web interface + inference engine
├── notebooks/          EDA and result analysis
├── logs/               Training logs + TensorBoard runs
├── evaluation/         Saved plots (ROC curve, confusion matrix)
├── config.yaml         Central configuration file
└── requirements.txt    Python dependencies
```

---

## Monitoring with TensorBoard
```bash
tensorboard --logdir logs/
```

---

## License

Distributed under the [MIT License](LICENSE).


# Made by:
**Chahiri Abderrahmane**
**Yahya Hadir**
**Hafsa Benabou**
**Ilhma Madihi**