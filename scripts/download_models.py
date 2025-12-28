import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Cấu hình: Mapping giữa thư mục đích và (Dataset ID, Tên file)
ARTIFACTS = {
    "models/teacher": [
        ("honghien123/ce-gate-pipeline-v3-3", "teacher.pt")
    ],
    "models/classifier": [
        ("honghien123/ArtefactModel-ce-gate-pipeline-v3", "final_model.pt"),
        ("honghien123/ArtefactModel-ce-gate-pipeline-v3", "final_best.pt")
    ],
    "data/interim": [
        ("honghien123/ce-gate-pipeline-v3-3", "ce_pairs_balanced.csv")
    ]
}

def download_and_extract():
    api = KaggleApi()
    api.authenticate() # Yêu cầu có file kaggle.json hoặc biến môi trường

    for folder, files in ARTIFACTS.items():
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(folder, exist_ok=True)
        
        for dataset_id, file_name in files:
            print(f"Downloading {file_name} from {dataset_id}...")
            try:
                # Tải file về thư mục đích
                api.dataset_download_file(dataset_id, file_name, path=folder)
                
                # Kaggle API thường tải về file .zip, cần giải nén nếu cần
                zip_path = os.path.join(folder, file_name + ".zip")
                if os.path.exists(zip_path):
                    print(f"Extracting {zip_path}...")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(folder)
                    os.remove(zip_path) # Xóa file zip cho nhẹ
                    print(f"Extracted to {folder}/{file_name}")
                else:
                    print(f"Downloaded to {folder}/{file_name}")
                    
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")

if __name__ == "__main__":
    print("Starting Artifact Download...")
    download_and_extract()
    print("All done! You are ready to run the models.")