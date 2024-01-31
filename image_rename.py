import os
import shutil
# 순서 정보를 추출하여 이미지를 순서대로 복사
# 파일 이름 정렬 필수

# 원본 이미지가 있는 디렉토리 경로
source_directory = "/home/ubuntu/cw/data/open/17places/17places/ref"

# 새로운 디렉토리 경로
destination_directory = "/home/ubuntu/cw/data/open/17places/17places/rename/ref"

# 디렉토리 내의 파일 목록을 얻어옵니다
file_list = os.listdir(source_directory)

# 파일 이름을 정렬합니다
file_list.sort()

# 이미지 파일 이름에서 순서 정보를 추출하여 순서대로 이름을 변경하고 복사
for index, filename in enumerate(file_list):
    if filename.endswith(".jpg"):
        new_filename = f"image_raw{index}.jpg"
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(destination_directory, new_filename)
        shutil.copy2(source_path, destination_path)

print(f"{len(file_list)}개의 이미지를 순서대로 이름을 변경하고 {destination_directory}에 복사했습니다.")
