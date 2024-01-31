import os
import shutil
import random

# 원본 이미지가 있는 디렉토리 경로
source_directory = "/home/ubuntu/cw/data/open/oxford/Oxford_Robotcar/oxDataPart/rename1"

# Query 이미지를 저장할 디렉토리 경로
query_directory = "/home/ubuntu/cw/data/open/oxford/Oxford_Robotcar/oxDataPart/split1/query"

# Reference 이미지를 저장할 디렉토리 경로
reference_directory = "/home/ubuntu/cw/data/open/oxford/Oxford_Robotcar/oxDataPart/split1/ref"

# 복사할 Query 이미지 개수
num_query_images = 60

# 디렉토리 내의 파일 목록을 얻어옵니다
file_list = os.listdir(source_directory)

# 무작위로 Query 이미지를 선택
query_image_indices = random.sample(range(len(file_list)), num_query_images)

# Query 이미지와 Reference 이미지로 나눠서 복사
for index in range(len(file_list)):
    source_filename = file_list[index]
    source_path = os.path.join(source_directory, source_filename)
    
    if index in query_image_indices:
        query_destination_path = os.path.join(query_directory, source_filename)
        shutil.copy2(source_path, query_destination_path)
    else:
        reference_destination_path = os.path.join(reference_directory, source_filename)
        shutil.copy2(source_path, reference_destination_path)

print(f"{num_query_images}개의 Query 이미지와 {len(file_list) - num_query_images}개의 Reference 이미지를 복사했습니다.")
