import sqlite3
import cv2
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from shapely.geometry import box
from shapely.geometry.polygon import Polygon

def compare_images(query_info, reference_info):
    total_area_diff = 0
    total_iou_score = 0

    if not query_info or not reference_info:
        return total_area_diff, total_iou_score

    min_length = min(len(query_info), len(reference_info))

    for i in range(min_length):
        query_bbox = query_info[i]['bbox']
        reference_bbox = reference_info[i]['bbox']

        area_diff = abs(query_bbox[2] * query_bbox[3] - reference_bbox[2] * reference_bbox[3])

        iou_score = calculate_iou(query_bbox, reference_bbox)

        total_area_diff += area_diff
        total_iou_score += iou_score

    return total_area_diff, total_iou_score

def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    bbox1_area = w1 * h1
    bbox2_area = w2 * h2
    union_area = bbox1_area + bbox2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def get_query_image_names(query_database_file):
    conn = sqlite3.connect(query_database_file)
    cursor = conn.cursor()
    cursor.execute("SELECT filename FROM images")
    query_image_names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return query_image_names

def histogram_matching(database_file, query_database_file, query_image_filenames):
    conn = sqlite3.connect(database_file)
    cursor = conn.cursor()

    query_conn = sqlite3.connect(query_database_file)
    query_cursor = query_conn.cursor()

    cursor.execute("SELECT filename, info_dict FROM images")
    reference_images = cursor.fetchall()

    results = []

    for query_filename in query_image_filenames:
        # Get query image info
        query_cursor.execute("SELECT info_dict FROM images WHERE filename=?", (query_filename,))
        query_info_dict = query_cursor.fetchone()[0]
        query_info = json.loads(query_info_dict)

        min_difference = float('inf')
        most_similar_image_filename = ""

        for ref_filename, reference_info_dict in reference_images:
            reference_info = json.loads(reference_info_dict)

            total_area_diff, total_iou_score = compare_images(query_info, reference_info)

            if total_area_diff < min_difference:
                min_difference = total_area_diff
                most_similar_semantic = query_info
                most_similar_image_filename = ref_filename

        query_image_id = int(''.join(filter(str.isdigit, query_filename)))
        most_similar_image_id = int(''.join(filter(str.isdigit, most_similar_image_filename)))
        id_difference = abs(query_image_id - most_similar_image_id)
        id_similar = id_difference <= 1
        results.append((most_similar_semantic, id_difference))

    conn.close()
    query_conn.close()
    return results

def prepare_data(hm_database_file, hm_query_database_file, query_image_names):
    matching_results = histogram_matching(hm_database_file, hm_query_database_file, query_image_names)
    features = []
    labels = []
    for hist_str, id_diff in matching_results:
        if hist_str:
            for obj in hist_str:
                area_feature = obj['area']
                bbox_feature = obj['bbox']
                features.append([area_feature, *bbox_feature])
                labels.append(1 if id_diff <= 1 else 0)

    return np.array(features), np.array(labels)

def main():
    hm_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference_ver4_edit4_noc.db"
    hm_query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query_ver4_edit4_noc.db"

    query_image_names = get_query_image_names(hm_query_database_file)
    matching_results = histogram_matching(hm_database_file, hm_query_database_file, query_image_names)

    X, y = prepare_data(hm_database_file, hm_query_database_file, query_image_names)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)    
    model = SVC(C=1, gamma=1, kernel="rbf")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    ml_model_accuracy = accuracy_score(y_test, predictions) * 100
    
    # 예측이 틀린 이미지의 정보 출력
    incorrect_predictions = np.where(predictions != y_test)[0]
    total_count = len(query_image_names)  # 전체 이미지 개수
    incorrect_count = len(incorrect_predictions)  # 틀린 이미지 개수
    
    print("\n")
    print(f"전체 쿼리 이미지 개수: {total_count}")
    print(f"틀린 이미지 개수: {incorrect_count}")
    print(f"ML 모델 정확도: {ml_model_accuracy:.2f}%")  # 수정된 부분
    
    # y_test와 predictions 출력
    print("\n")
    print("y_test:", y_test)
    print("predictions:", predictions)
    
if __name__ == "__main__":
    main()
