# MELD-Only Local Test Plan

Bu plan, `kalemdarrr/multimodel_emotion_recognition` projesinde ilk deneme olarak yalnızca **MELD** veri setiyle yerel bilgisayarda çalışmak için hazırlanmıştır.

Amaç:

```text
Önce sadece MELD ile pipeline'ın çalıştığını görmek.
Sonra mümkünse MELD-only baseline eğitimi almak.
RAVDESS, CREMA-D, MOSEI, IEMOCAP, GoEmotions ve image-only veri setleri bu aşamada kullanılmayacak.
```

---

## 1. Bu aşamada yapılacaklar

```text
1. Yerel ortamı kontrol et.
2. MELD veri setini klasöre yerleştir.
3. MELD için manifest hazırlama scripti oluştur.
4. train / val / test manifestlerini üret.
5. Manifestleri doğrula.
6. Küçük bir subset ile smoke test yap.
7. Audio feature extraction çalıştır.
8. Video feature extraction çalıştır.
9. Donanım yeterliyse MELD-only training çalıştır.
10. Test/evaluation sonucu al.
```

---

## 2. Bu aşamada yapılmayacaklar

```text
- RAVDESS eklenmeyecek.
- CREMA-D eklenmeyecek.
- MOSEI eklenmeyecek.
- GoEmotions eklenmeyecek.
- FER2013 / AffectNet / RAF-DB eklenmeyecek.
- Label harmonization karmaşıklaştırılmayacak.
- Combined manifest oluşturulmayacak.
```

Bu ilk aşamanın amacı veri çeşitlendirmek değil, **MELD ile çalışan sağlam bir baseline** elde etmektir.

---

## 3. Hedef label seti

Projenin hedef label seti:

```text
neutral, joy, sadness, anger, fear, disgust, surprise
```

MELD zaten bu label setine uyumludur.

Mapping:

| MELD label | Proje label | Güven |
|---|---|---|
| neutral | neutral | high |
| joy | joy | high |
| sadness | sadness | high |
| anger | anger | high |
| fear | fear | high |
| disgust | disgust | high |
| surprise | surprise | high |

Bu aşamada label mapping karmaşık olmayacak.

---

## 4. Önerilen yerel klasör yapısı

Repo kökünde:

```text
multimodel_emotion_recognition/
  data/
    raw/
      meld/
        train_sent_emo.csv
        dev_sent_emo.csv
        test_sent_emo.csv
        train_splits/
        dev_splits_complete/
        output_repeated_splits_test/
    processed/
      meld/
        train_raw.jsonl
        val_raw.jsonl
        test_raw.jsonl
        train_audio.jsonl
        val_audio.jsonl
        test_audio.jsonl
        train.jsonl
        val.jsonl
        test.jsonl
    features/
      audio/
        meld/
          train/
          val/
          test/
      video/
        meld/
          train/
          val/
          test/
  artifacts/
    runs/
      meld_baseline/
    evals/
      meld_baseline_test/
  docs/
    meld_local_plan.md
```

Not:

MELD klasör adları indirilen pakete göre küçük farklılık gösterebilir. Script pathleri argüman olarak almalı; klasör adını hard-code etmemeli.

---

## 5. Yerel ortam kontrolü

### 5.1 Python ortamı

Öneri:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Paket kurulumu:

```bash
pip install -r requirements.txt
```

Eksik paket çıkarsa:

```bash
pip install transformers torch torchvision torchaudio librosa soundfile opencv-python-headless scikit-learn tqdm numpy pandas
```

### 5.2 GPU kontrolü

Python ile:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Karar:

```text
CUDA varsa:
  audio/video feature extraction ve training yerelde denenebilir.

CUDA yoksa:
  manifest hazırlama yapılır.
  küçük subset ile CPU smoke test yapılabilir.
  full training çok yavaş olabilir.
```

---

## 6. Oluşturulacak scriptler

Bu MELD-only aşamada yalnızca aşağıdaki scriptler gerekli:

```text
scripts/prepare_meld_manifest.py
scripts/report_manifest_stats.py
scripts/create_manifest_subset.py
```

Opsiyonel:

```text
docs/meld_local_usage.md
```

Bu aşamada gerek yok:

```text
prepare_ravdess_manifest.py
prepare_cremad_manifest.py
build_combined_manifest.py
prepare_mosei_manifest.py
```

---

## 7. `prepare_meld_manifest.py` gereksinimleri

Script görevi:

```text
MELD CSV + video/audio klasörlerinden repo formatına uygun JSONL manifest üretmek.
```

Argümanlar:

```bash
python scripts/prepare_meld_manifest.py \
  --csv data/raw/meld/train_sent_emo.csv \
  --video-root data/raw/meld/train_splits \
  --audio-root data/raw/meld/audio/train \
  --output data/processed/meld/train_raw.jsonl \
  --split train
```

`--audio-root` opsiyonel olmalı. Eğer ayrı audio yoksa video dosyasından ses çıkarılacaksa script yalnızca `video_path` yazabilir.

Üretilecek JSONL satırı örneği:

```json
{
  "sample_id": "meld_train_dia0_utt0",
  "label": "neutral",
  "text": "also I was the point person on my company's transition from the KL-5 to GR-6 system.",
  "group_id": "dia0",
  "speaker_id": "Chandler",
  "audio_path": "",
  "video_path": "data/raw/meld/train_splits/dia0_utt0.mp4",
  "metadata": {
    "source": "MELD",
    "split": "train",
    "dialogue_id": "0",
    "utterance_id": "0",
    "original_label": "neutral",
    "mapped_label": "neutral",
    "mapping_confidence": "high",
    "modality": "text_video"
  }
}
```

Zorunlu davranışlar:

```text
- CSV kolon adlarını esnek yakala:
  Dialogue_ID / dialogue_id
  Utterance_ID / utterance_id
  Speaker / speaker
  Utterance / utterance
  Emotion / emotion
- Label değerlerini lowercase normalize et.
- Sadece hedef label setindeki örnekleri al.
- Boş text varsa uyarı ver.
- Video dosyası bulunamazsa satırı yine yaz ama metadata.missing_video = true ekle.
- sample_id benzersiz olsun.
```

---

## 8. `report_manifest_stats.py` gereksinimleri

Script görevi:

```text
Manifest dosyasını okuyup temel veri kalitesi raporu üretmek.
```

Komut:

```bash
python scripts/report_manifest_stats.py \
  --manifest data/processed/meld/train_raw.jsonl \
  --output artifacts/evals/meld_train_raw_stats.json
```

Rapor içeriği:

```text
num_samples
label_counts
empty_text_count
missing_video_path
missing_audio_path
missing_audio_features
missing_video_features
duplicate_sample_ids
source_counts
split_counts
```

Amaç:

```text
Feature extraction veya training öncesinde verinin doğru üretildiğini görmek.
```

---

## 9. `create_manifest_subset.py` gereksinimleri

Script görevi:

```text
Küçük smoke test için manifestten az sayıda örnek seçmek.
```

Komut:

```bash
python scripts/create_manifest_subset.py \
  --input data/processed/meld/train_raw.jsonl \
  --output data/processed/meld/train_raw_small.jsonl \
  --max-per-label 5 \
  --seed 42
```

Aynısı val için:

```bash
python scripts/create_manifest_subset.py \
  --input data/processed/meld/val_raw.jsonl \
  --output data/processed/meld/val_raw_small.jsonl \
  --max-per-label 3 \
  --seed 42
```

Amaç:

```text
Tüm MELD'i işlemeye başlamadan önce pipeline'ın çalıştığını test etmek.
```

---

## 10. Manifest üretme komutları

Train:

```bash
python scripts/prepare_meld_manifest.py \
  --csv data/raw/meld/train_sent_emo.csv \
  --video-root data/raw/meld/train_splits \
  --output data/processed/meld/train_raw.jsonl \
  --split train
```

Val/dev:

```bash
python scripts/prepare_meld_manifest.py \
  --csv data/raw/meld/dev_sent_emo.csv \
  --video-root data/raw/meld/dev_splits_complete \
  --output data/processed/meld/val_raw.jsonl \
  --split val
```

Test:

```bash
python scripts/prepare_meld_manifest.py \
  --csv data/raw/meld/test_sent_emo.csv \
  --video-root data/raw/meld/output_repeated_splits_test \
  --output data/processed/meld/test_raw.jsonl \
  --split test
```

Not:

Eğer video klasör adları farklıysa komutlardaki `--video-root` pathleri değiştirilecek.

---

## 11. Smoke subset üretme

```bash
python scripts/create_manifest_subset.py \
  --input data/processed/meld/train_raw.jsonl \
  --output data/processed/meld/train_raw_small.jsonl \
  --max-per-label 5 \
  --seed 42
```

```bash
python scripts/create_manifest_subset.py \
  --input data/processed/meld/val_raw.jsonl \
  --output data/processed/meld/val_raw_small.jsonl \
  --max-per-label 3 \
  --seed 42
```

---

## 12. Manifest stats kontrolü

```bash
python scripts/report_manifest_stats.py \
  --manifest data/processed/meld/train_raw.jsonl \
  --output artifacts/evals/meld_train_raw_stats.json
```

```bash
python scripts/report_manifest_stats.py \
  --manifest data/processed/meld/val_raw.jsonl \
  --output artifacts/evals/meld_val_raw_stats.json
```

```bash
python scripts/report_manifest_stats.py \
  --manifest data/processed/meld/test_raw.jsonl \
  --output artifacts/evals/meld_test_raw_stats.json
```

---

## 13. Feature extraction — önce küçük subset

Önce küçük subset ile dene.

### 13.1 Audio feature extraction — small

CUDA varsa:

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/train_raw_small.jsonl \
  --output-manifest data/processed/meld/train_audio_small.jsonl \
  --feature-dir data/features/audio/meld/train_small \
  --device cuda
```

CPU varsa:

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/train_raw_small.jsonl \
  --output-manifest data/processed/meld/train_audio_small.jsonl \
  --feature-dir data/features/audio/meld/train_small \
  --device cpu
```

### 13.2 Video feature extraction — small

CUDA varsa:

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/train_audio_small.jsonl \
  --output-manifest data/processed/meld/train_small.jsonl \
  --feature-dir data/features/video/meld/train_small \
  --num-frames 8 \
  --device cuda
```

CPU varsa:

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/train_audio_small.jsonl \
  --output-manifest data/processed/meld/train_small.jsonl \
  --feature-dir data/features/video/meld/train_small \
  --num-frames 8 \
  --device cpu
```

Val için de aynı küçük akış yapılmalı.

---

## 14. Feature extraction — full MELD

Small subset sorunsuz çalışırsa full MELD'e geç.

### Train audio

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/train_raw.jsonl \
  --output-manifest data/processed/meld/train_audio.jsonl \
  --feature-dir data/features/audio/meld/train \
  --device cuda
```

### Train video

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/train_audio.jsonl \
  --output-manifest data/processed/meld/train.jsonl \
  --feature-dir data/features/video/meld/train \
  --num-frames 8 \
  --device cuda
```

Val ve test için aynı işlem:

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/val_raw.jsonl \
  --output-manifest data/processed/meld/val_audio.jsonl \
  --feature-dir data/features/audio/meld/val \
  --device cuda
```

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/val_audio.jsonl \
  --output-manifest data/processed/meld/val.jsonl \
  --feature-dir data/features/video/meld/val \
  --num-frames 8 \
  --device cuda
```

```bash
PYTHONPATH=src python scripts/extract_audio_features.py \
  --input-manifest data/processed/meld/test_raw.jsonl \
  --output-manifest data/processed/meld/test_audio.jsonl \
  --feature-dir data/features/audio/meld/test \
  --device cuda
```

```bash
PYTHONPATH=src python scripts/extract_video_features.py \
  --input-manifest data/processed/meld/test_audio.jsonl \
  --output-manifest data/processed/meld/test.jsonl \
  --feature-dir data/features/video/meld/test \
  --num-frames 8 \
  --device cuda
```

CPU kullanılıyorsa `--device cuda` yerine `--device cpu` yaz.

---

## 15. Final manifest validation

```bash
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/processed/meld/train.jsonl \
  --labels neutral joy sadness anger fear disgust surprise
```

```bash
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/processed/meld/val.jsonl \
  --labels neutral joy sadness anger fear disgust surprise
```

```bash
PYTHONPATH=src python scripts/validate_manifest.py \
  --manifest data/processed/meld/test.jsonl \
  --labels neutral joy sadness anger fear disgust surprise
```

---

## 16. Yerel eğitim kararı

### Eğer NVIDIA GPU varsa

MELD-only baseline denenebilir.

Önerilen başlangıç:

```text
batch_size: 4 veya 8
epochs: 3 ile hızlı test
max_text_length: 128
num_frames: 8
```

Önce küçük training smoke test:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/default.json \
  --train-manifest data/processed/meld/train_small.jsonl \
  --val-manifest data/processed/meld/val_small.jsonl \
  --output-dir artifacts/runs/meld_small_smoke
```

Sonra full MELD:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/default.json \
  --train-manifest data/processed/meld/train.jsonl \
  --val-manifest data/processed/meld/val.jsonl \
  --output-dir artifacts/runs/meld_baseline
```

### Eğer sadece CPU varsa

Tam eğitim önerilmez.

Yapılabilecekler:

```text
- manifest üret
- stats raporu al
- small subset feature extraction dene
- small subset ile train.py çalışıyor mu test et
```

Full MELD eğitimi Colab'a bırakılmalı.

---

## 17. Evaluation

Eğer training tamamlanırsa:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/default.json \
  --checkpoint artifacts/runs/meld_baseline/best_model.pt \
  --manifest data/processed/meld/test.jsonl \
  --output-dir artifacts/evals/meld_baseline_test
```

Bakılacak metrikler:

```text
accuracy
macro_f1
weighted_f1
per_class_f1
confusion_matrix
```

Özellikle:

```text
fear
disgust
surprise
```

---

## 18. Başarı kriteri

Bu aşamanın başarılı sayılması için:

```text
[ ] MELD train/val/test raw manifestleri üretildi.
[ ] Label dağılımı mantıklı görünüyor.
[ ] Küçük subset üretildi.
[ ] Small subset audio/video feature extraction çalıştı.
[ ] Full feature extraction mümkünse tamamlandı.
[ ] Manifest validation geçti.
[ ] Small train smoke test çalıştı.
[ ] GPU yeterliyse MELD baseline training tamamlandı.
[ ] Evaluation sonucu alındı.
```

---

## 19. Sorun çıkarsa karar ağacı

### Video path bulunamıyor

```text
- MELD video dosya adını kontrol et.
- Genelde dia{Dialogue_ID}_utt{Utterance_ID}.mp4 formatı kullanılır.
- prepare script path fallback denemeli.
```

### Audio path yok

```text
- Eğer ayrı audio çıkarılmadıysa sorun değil.
- Audio extractor video dosyasından ses okuyamıyorsa ffmpeg ile audio çıkarma adımı gerekebilir.
```

Opsiyonel audio extraction:

```bash
ffmpeg -i input.mp4 -vn -ac 1 -ar 16000 output.wav
```

### CUDA out of memory

```text
- batch_size düşür.
- num_frames 8 kalsın veya 4'e düşür.
- max_text_length 128 kullan.
- small subset ile test et.
```

### CPU çok yavaş

```text
- Full feature extraction ve full training Colab'a taşınmalı.
```

---

## 20. Sonraki aşama

MELD-only baseline elde edilmeden diğer veri setlerine geçilmemeli.

Sonraki aşama:

```text
1. RAVDESS ekle.
2. CREMA-D ekle.
3. MELD test setinde karşılaştır.
4. İyileşme yoksa class-based augmentation stratejisine geç.
```
