import json
from pathlib import Path

def load_json_annotations(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data['annotations']

def get_files_list(directory, extension='*.jpg'):
    path = Path(directory)
    return set(file.name for file in path.glob(extension))

def get_referenced_files_from_json(annotations):
    return set(anno['file_name'].replace('.png', '.jpg') for anno in annotations)

def compare_files(referenced_files, actual_files):
    missing_from_actual = referenced_files - actual_files
    missing_from_json = actual_files - referenced_files
    return missing_from_actual, missing_from_json

# 경로 설정
json_path = '/mnt/sda/suhohan/coco2017/annotations/panoptic_train2017.json'
images_dir = '/mnt/sda/suhohan/coco2017/train2017'

# 데이터 로드
annotations = load_json_annotations(json_path)
actual_images = get_files_list(images_dir, '*.jpg')
json_referenced_images = get_referenced_files_from_json(annotations)

# 파일 목록 비교
missing_images_from_actual, missing_images_from_json = compare_files(json_referenced_images, actual_images)

# 결과 출력
print("Missing image files in actual directory:", len(missing_images_from_actual))
print("Extra image files in actual directory:", len(missing_images_from_json))
