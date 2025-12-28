#!/bin/bash

# ==============================================================================
# Script: download_artifacts.sh
# Mô tả: Tự động tải Trọng số mô hình (Weights) và Dữ liệu cấu hình từ Kaggle
# Yêu cầu: Đã cài đặt thư viện 'kaggle' (pip install kaggle) và có file kaggle.json
# ==============================================================================

# Dừng script ngay lập tức nếu có lệnh bị lỗi
set -e

# Định nghĩa ID của các Dataset trên Kaggle
DATASET_CE_GATE="honghien123/ce-gate-pipeline-v3-3" # https://www.kaggle.com/datasets/honghien123/ce-gate-pipeline-v3-3
DATASET_CLASSIFIER="honghien123/ArtefactModel-ce-gate-pipeline-v3" # https://www.kaggle.com/datasets/honghien123/artefactmodel-ce-gate-pipeline-v3

echo " Bắt đầu quá trình tải Artifacts..."

# 1. Kiểm tra xem Kaggle CLI đã được cài chưa
if ! command -v kaggle &> /dev/null; then
    echo " Lỗi: Không tìm thấy lệnh 'kaggle'. Vui lòng chạy: pip install kaggle"
    exit 1
fi

# ==============================================================================
# GIAI ĐOẠN 1: TẢI CE TEACHER & CONFIGS (Stage 1)
# ==============================================================================
echo "----------------------------------------------------------------"
echo " Đang tải Artifacts cho Giai đoạn 1 (CE Gate)..."

# Tạo thư mục đích
mkdir -p models/teacher
mkdir -p configs
mkdir -p data/interim

# 1.1 Tải Teacher Model (teacher.pt)
echo "   -> Tải teacher.pt (1GB)..."
kaggle datasets download -d $DATASET_CE_GATE -f teacher.pt -p models/teacher --force

# 1.2 Tải & Đổi tên Configs (temp.json -> ce_temp.json)
echo "   -> Tải cấu hình nhiệt độ (ce_temp.json)..."
kaggle datasets download -d $DATASET_CE_GATE -f temp.json -p configs --force
mv configs/temp.json configs/ce_temp.json

# 1.3 Tải Label Mapping
echo "   -> Tải labels.json..."
kaggle datasets download -d $DATASET_CE_GATE -f labels.json -p configs --force

# 1.4 (Tùy chọn) Tải dữ liệu huấn luyện Teacher nếu cần tái lập training
# echo "   -> Tải ce_pairs_balanced.csv..."
# kaggle datasets download -d $DATASET_CE_GATE -f ce_pairs_balanced.csv -p data/interim --force

# ==============================================================================
# GIAI ĐOẠN 2: TẢI CLASSIFIER MODEL (Stage 2)
# ==============================================================================
echo "----------------------------------------------------------------"
echo " Đang tải Artifacts cho Giai đoạn 2 (Final Classifier)..."

# Tạo thư mục đích
mkdir -p models/classifier

# 2.1 Tải Final Model
echo "   -> Tải final_model.pt (1GB)..."
kaggle datasets download -d $DATASET_CLASSIFIER -f final_model.pt -p models/classifier --force

# 2.2 Tải Best Model (Optional)
echo "   -> Tải final_best.pt (1GB)..."
kaggle datasets download -d $DATASET_CLASSIFIER -f final_best.pt -p models/classifier --force

# 2.3 Tải Configs Classifier
echo "   -> Tải cls_temp.json & final_meta.json..."
kaggle datasets download -d $DATASET_CLASSIFIER -f cls_temp.json -p configs --force
kaggle datasets download -d $DATASET_CLASSIFIER -f final_meta.json -p configs --force

# ==============================================================================
# XỬ LÝ FILE ZIP (Nếu Kaggle tự động nén file lẻ)
# ==============================================================================
echo "----------------------------------------------------------------"
echo "📦 Đang kiểm tra và giải nén (nếu cần)..."

# Hàm giải nén và xóa file zip
extract_if_needed() {
    DIR=$1
    FILE=$2
    if [ -f "$DIR/$FILE.zip" ]; then
        echo "   -> Giải nén $FILE.zip..."
        unzip -o -q "$DIR/$FILE.zip" -d "$DIR"
        rm "$DIR/$FILE.zip"
    fi
}

extract_if_needed "models/teacher" "teacher.pt"
extract_if_needed "models/classifier" "final_model.pt"
extract_if_needed "models/classifier" "final_best.pt"

echo "----------------------------------------------------------------"
echo "   HOÀN TẤT! Tất cả model và dữ liệu đã sẵn sàng."
echo "   - Teacher: models/teacher/teacher.pt"
echo "   - Classifier: models/classifier/final_model.pt"
echo "   - Configs: configs/"