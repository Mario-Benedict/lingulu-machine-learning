# 🐯 Lingulu — Pronunciation Scoring Model (Machine Learning)

This module contains the Machine Learning pipeline used in **Lingulu** to evaluate English pronunciation using speech recognition and phoneme-level scoring.

The model is based on **Wav2Vec2**, fine-tuned for pronunciation assessment using a **phoneme vocabulary** and **GOP (Goodness of Pronunciation) scoring**.

---

## 🎯 Purpose

The goal of this model is **not** to transcribe speech into text, but to:

> Evaluate how accurately a user pronounces words and sentences at the phoneme level.

This enables Lingulu to provide **objective pronunciation feedback** to learners.

---

## 🧠 Core Concept

The pipeline works as follows:

1. User speaks a sentence
2. Audio is processed by **fine-tuned Wav2Vec2**
3. Model predicts **phoneme sequence probabilities**
4. GOP score is calculated per phoneme
5. Scores are aggregated into:
   - Phoneme score
   - Word score
   - Sentence pronunciation score
6. Results are sent to backend as evaluation feedback

---

## 🏗️ Model Architecture

| Component | Description |
|-----------|-------------|
| Base Model | Facebook Wav2Vec2 |
| Fine-tuning Target | Phoneme recognition (not text ASR) |
| Vocabulary | Custom phoneme set (ARPAbet/IPA-based) |
| Output | Phoneme probability distribution |
| Scoring Method | GOP (Goodness of Pronunciation) |

---
## 📂 Folder Structure
```
lingulu-machine-learning/
│── notebooks/
│   ├── models/
│   │   ├── v1/
|   |   |  ├── model_finetune.ipynb
|   |   |  └── train_history.csv
│   │   ├── v2/
|   |   |  ├── model_finetune.ipynb
|   |   |  └── train_history.csv
│   │   └── v3/
|   |      ├── model_finetune.ipynb
|   |      └── train_history.csv
│   ├── audio_converter.ipynb
│   ├── dataset_sampling.ipynb
│   └── model_evaluate_v3.ipynb
│── .gitignore
└── requirements.txt
```
---
## ⚙️ Installation

### Requirements

- Python 3.10+
- PyTorch
- Transformers (HuggingFace)
- Librosa
- NumPy

### Install dependencies:

```bash
pip install -r requirements.txt
```

### 🧪 Training the Model

```bash
python train.py
```

### 🎙️ Inference (Pronunciation Evaluation)

```bash
python infer.py --audio sample.wav --text "hello world"
```

### Output 
```yaml

```
## 🧮 GOP Scoring

GOP measures how closely a spoken phoneme matches the expected phoneme.

Formula : 

```lua
GOP(p) = log P(p | audio) - max log P(q | audio)
```
Where:
- p = expected phoneme
- q = all possible phonemes

Higher GOP = better pronunciation.

---
## 🔗 Integration with Backend

Input:

- Audio file from user

- Expected sentence

Output (JSON):

```json

```

---

## 📊 Why Wav2Vec2?

Wav2Vec2 is used because:

- Strong representation of speech features

- Works well with limited labeled data

- Suitable for phoneme-level tasks

- State-of-the-art for speech understanding

---

## 🚀 Future Improvements

- Support IPA phoneme set

- Noise-robust training

- Real-time scoring

- Accent adaptation

---

Made with love ❤️, lack of sleep 🥱 and tears 💧 by MACAN MULAZ 🐅
