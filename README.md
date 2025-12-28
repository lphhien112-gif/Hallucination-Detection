
# UIT Data Science Challenge 2025: Phát hiện Ảo giác trong LLM Tiếng Việt

**Team Name:** [Điền Tên Đội Của Bạn]

**Track:** Hallucination Detection

## 📖 Tổng quan (Overview)

Repository này chứa mã nguồn giải pháp chính thức cho cuộc thi **UIT Data Science Challenge 2025**. Mục tiêu của dự án là xây dựng hệ thống tự động phát hiện ảo giác (Hallucination) trong các mô hình ngôn ngữ lớn tiếng Việt, phân loại đầu ra thành 3 nhãn:

1. **No Hallucination (0):** Phản hồi chính xác, thông tin hoàn toàn dựa trên ngữ cảnh được cung cấp.
2. **Intrinsic Hallucination (1):** Phản hồi mâu thuẫn hoặc bóp méo thông tin so với ngữ cảnh.
3. **Extrinsic Hallucination (2):** Phản hồi chứa thông tin bổ sung không có căn cứ trong ngữ cảnh.

Chúng tôi đề xuất phương pháp **Two-Stage Pipeline** (Quy trình 2 giai đoạn) kết hợp giữa trích xuất minh chứng (Evidence Extraction) bằng nhãn yếu và phân loại ngữ nghĩa (Semantic Classification) với cơ chế Attention Pooling.

## 🚀 Phương pháp tiếp cận (Methodology)

Giải pháp được chia làm 2 giai đoạn chính để xử lý vấn đề ngữ cảnh dài (Long Context) và nhiễu:

### Giai đoạn 1: CE Gate Pipeline (Trích xuất minh chứng)

Thay vì đưa toàn bộ ngữ cảnh (Context) vào mô hình phân loại, chúng tôi lọc ra các đoạn văn quan trọng nhất.

* **Teacher Model:** Sử dụng `xlm-roberta-base` làm Cross-Encoder để chấm điểm sự liên quan giữa (Prompt + Response) và từng phân đoạn (Span) trong Context.
* **Weak Supervision:** Sử dụng các luật heuristic (khớp số liệu, từ phủ định, overlap) để tạo nhãn giả lập huấn luyện Teacher.
* **Dual-Beam Packer:** Thuật toán chọn lọc minh chứng thông minh, cân bằng giữa thông tin **Ủng hộ** (Support) và thông tin **Mâu thuẫn** (Conflict) để đảm bảo mô hình nhìn thấy được cả hai khía cạnh của ảo giác.

### Giai đoạn 2: Final Classifier (Phân loại cuối cùng)

* **Backbone:** Sử dụng `vinai/phobert-large` (State-of-the-art cho tiếng Việt).
* **Attention Pooling:** Thay vì dùng token `[CLS]`, chúng tôi dùng vector của (Prompt + Response) làm *Query* để "chú ý" (attend) vào các token quan trọng trong Minh chứng (*Key/Value*).
* **Feature Fusion:** Kết hợp vector văn bản với các chỉ số ngữ nghĩa (`support_mass`, `conflict_mass`) từ Giai đoạn 1.
* **Training Tricks:** Weighted Focal Loss (xử lý mất cân bằng mẫu), R-Drop, FGM (Adversarial Training).

## 📂 Cấu trúc dự án (Project Structure)

```text
uit-dsc-hallucination-detection/
├── configs/                        # Chứa các file tham số cấu hình và mapping
│   ├── labels.json                 # Từ output ce_gate (Label mapping)
│   ├── ce_temp.json                # Từ output ce_gate (temp.json - đổi tên cho rõ)
│   ├── cls_temp.json               # Từ output classifier
│   └── final_meta.json             # Từ output classifier (Hyperparams)
│
├── data/                           # Quản lý dữ liệu (Dùng .gitignore cho file lớn)
│   ├── raw/                        # Dữ liệu gốc cuộc thi (vihallu-train.csv...)
│   ├── interim/                    # Dữ liệu trung gian (Stage 1 tạo ra)
│   │   ├── ce_pairs_balanced.csv   # Dùng huấn luyện Teacher
│   │   ├── train_v3_semantic.csv   # Phân tích ngữ nghĩa
│   │   └── val_v3_semantic.csv
│   │
│   └── processed/                  # Dữ liệu đã xử lý quan trọng (Input cho Stage 2)
│       ├── hybrid_train_v3_coverage_with_mass.csv  
│       ├── hybrid_val_v3_coverage_with_mass.csv    
│       └── hybrid_test_v3_coverage_with_mass.csv   
|
├── models/                         # Nơi chứa trọng số (Không push lên Git, chỉ lưu link)
│   ├── teacher/
│   │   └── teacher.pt              # [Tải từ Kaggle Dataset 1](https://www.kaggle.com/datasets/honghien123/ce-gate-pipeline-v3-3)
│   └── classifier/
│       ├── final_model.pt          # [Tải từ Kaggle Dataset 2](https://www.kaggle.com/datasets/honghien123/artefactmodel-ce-gate-pipeline-v3)
│       └── final_best.pt           # [Tải từ Kaggle Dataset 2](https://www.kaggle.com/datasets/honghien123/artefactmodel-ce-gate-pipeline-v3)
|
├── notebooks/                      # Mã nguồn chạy thử nghiệm
│   ├── 01_ce_gate_pipeline.ipynb
│   └── 02_final_classifier.ipynb
│
├── reports/                        # Báo cáo và chỉ số đánh giá
│   ├── gate_report.json            # Từ output ce_gate
│   ├── evaluation_metrics.json     # Từ output classifier
│   └── MANIFEST.json               # Danh sách artifact
│
├── submissions/                    # Kết quả nộp bài
│   ├── submit.csv                  # Kết quả cuối cùng (có temp scaling)
│   ├── submit_no_temp.csv          # Kết quả tham khảo (không temp)
│   └── archive/                    # Lưu các file .zip
│       ├── submit.zip
│       └── submit_no_temp.zip
│
├── scripts/                        # Scripts tiện ích
│   └── download_artifacts.sh       # Script tải model từ Kaggle về folder models/
│
├── .gitignore                      # Chặn file nặng
├── README.md                       # Hướng dẫn dự án
└── requirements.txt                # Thư viện cần thiết

```

## 🛠️ Cài đặt & Hướng dẫn sử dụng

### 1. Chuẩn bị môi trường

Yêu cầu Python 3.10+ và PyTorch có hỗ trợ CUDA.

```bash
# Clone repository
git clone https://github.com/[username]/uit-dsc-hallucination-detection.git
cd uit-dsc-hallucination-detection

# Cài đặt thư viện
pip install -r requirements.txt

```

### 2. Tải Dữ liệu & Trọng số Mô hình

Do giới hạn dung lượng của GitHub, các file trọng số mô hình (`.pt` ~3GB) và dữ liệu huấn luyện lớn được lưu trữ trên Kaggle. Bạn cần chạy script sau để tải chúng về đúng thư mục:

```bash
# Cấp quyền thực thi (nếu cần)
chmod +x scripts/download_artifacts.sh

# Chạy script tải dữ liệu
python scripts/download_models.py
```

*Lưu ý: Cần cấu hình Kaggle API Key (`~/.kaggle/kaggle.json`) để script hoạt động.*

### 3. Quy trình Huấn luyện & Inference

**Bước 1: Chạy Giai đoạn 1 (Evidence Extraction)**
Mở notebook `notebooks/01_ce_gate_pipeline.ipynb`.

* **Input:** `data/raw/vihallu-train.csv`
* **Output:** Dữ liệu đã lọc minh chứng tại `data/processed/hybrid_train_v3_coverage_with_mass.csv`.

**Bước 2: Chạy Giai đoạn 2 (Training & Prediction)**
Mở notebook `notebooks/02_final_classifier.ipynb`.

* **Input:** Dữ liệu processed từ Bước 1.
* **Output:** Model `final_model.pt` và file nộp bài `submissions/submit.csv`.

## 📊 Kết quả (Results)

Hiệu suất mô hình trên tập Validation (được trích xuất từ `evaluation_metrics.json`):

| Metric | Score | Ghi chú |
| --- | --- | --- |
| **Final Macro F1** | **0.880** | Stage 2 Classifier |
| **Accuracy** | **0.879** | Stage 2 Classifier |
| **Teacher F1** | 0.974 | Stage 1 (Weak-label task) |

**Chi tiết từng lớp (Per-class Performance):**

| Class | Precision | Recall | F1-Score |
| --- | --- | --- | --- |
| **No Hallucination** | 0.97 | 0.96 | **0.96** |
| **Intrinsic** | 0.85 | 0.85 | 0.85 |
| **Extrinsic** | 0.82 | 0.83 | 0.82 |

## 📜 Giấy phép

Mã nguồn được phân phối dưới giấy phép **MIT**. Dữ liệu và các trọng số mô hình tuân theo giấy phép **CC-BY-SA 4.0** theo quy định của cuộc thi UIT Data Science Challenge 2025.
