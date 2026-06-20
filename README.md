# CV Screening Entity Recognition (NER) System

Proyek ini bertujuan untuk mengekstraksi entitas kunci dari dokumen resume secara otomatis menggunakan berbagai pendekatan Natural Language Processing (NLP), mulai dari sistem berbasis aturan (*rule-based*) hingga model Deep Learning berbasis Transformer (RoBERTa).

## 🚀 Fitur Utama
- **Multi-Model Comparison**: Membandingkan 4 arsitektur berbeda untuk ekstraksi entitas.
- **Entity Extraction**: Mengekstrak 10 kategori entitas termasuk: `Name`, `Designation`, `Companies worked at`, `Skills`, `Email Address`, `College Name`, `Degree`, `Location`, `Graduation Year`, dan `Years of Experience`.
- **Custom Pipeline**: Menggunakan spaCy v3 dan Hugging Face Transformers.

## 📊 Perbandingan Performa Model
Berdasarkan pengujian pada data uji, berikut adalah ringkasan hasil evaluasi untuk model-model utama:

| Model Arsitektur           | Precision | Recall | F1-Score |
| :------------------------- | :-------: | :----: | :------: |
| **Model 1: Rule-Based**    | 0.1049    | 0.3064 | 0.1348   |
| **Model 2: Statistical CNN** | 0.5253    | 0.3985 | 0.4457   |
| **Model 3: DistilBERT**    | 0.2330    | 0.1950 | 0.2120   |
| **Model 4: RoBERTa**       | 0.4989    | 0.5534 | **0.5248** |

### Insight Utama:
*   **Akurasi Tertinggi**: **RoBERTa** memberikan hasil yang paling seimbang (F1: 0.52).
*   **Entitas Spesifik**: Model CNN (spaCy) sangat unggul dalam mendeteksi **Name** (F1: 0.86) dan **Email Address** (F1: 0.73).
*   **Kelemahan**: Semua model masih kesulitan mendeteksi entitas `Years of Experience` secara akurat.

## Dataset:
https://github.com/laxmimerit/CV-Parsing-using-Spacy-3

## 📁 Struktur Repositori
- `data/`: Berisi dataset `train.json` dan `test.json`.
- `notebooks/`: Langkah-langkah pengembangan mulai dari EDA hingga training model.
- `results/`: File JSON hasil evaluasi tiap model.

## 🛠️ Cara Instalasi
1. Clone repositori ini:
   ```bash
   git clone [link removed]
2. Pastikan DATA_PATH pada setiap notebook