from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import h5py
import numpy as np
from pathlib import Path
from pprint import pformat
from hloc import extract_features, match_features
import sqlite3
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import time
import json

# Hloc NNsearch
def nearest_neighbor_search(query_descriptor, db_global_descriptors):
    distances = np.linalg.norm(db_global_descriptors - query_descriptor, axis=1)
    most_similar_image_idx = np.argmin(distances)
    return most_similar_image_idx, distances[most_similar_image_idx]

# Histogram BBox 계산
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

def prepare_data(db_data, query_db_data, hm_database_file, hm_query_database_file, query_image_names, db_global_descriptors, query_global_descriptors):
    features = []
    labels = []

    for query_index, query_name in enumerate(query_image_names):
        query_descriptor = query_global_descriptors[query_index]
        most_similar_image_idx, _ = nearest_neighbor_search(query_descriptor, db_global_descriptors)

        db_image_name = list(db_data.keys())[most_similar_image_idx]
        hloc_id_difference = abs(int(''.join(filter(str.isdigit, query_name))) - 
                                 int(''.join(filter(str.isdigit, db_image_name))))
        hloc_label = 1 if hloc_id_difference <= 1 else 0

        matching_results = histogram_matching(hm_database_file, hm_query_database_file, [query_name])
        histogram_array = []
        histogram_label = []
        for hist_str, id_diff in matching_results:
            if hist_str:
                for obj in hist_str:
                    area_feature = obj['area']
                    bbox_feature = obj['bbox']
                    histogram_array.append([area_feature, *bbox_feature])
                    histogram_label.append(1 if id_diff <= 1 else 0)

        # hloc_label과 histogram_label 중 하나라도 1이면 labels에 1 추가, 그렇지 않으면 0 추가
        labels.append(1 if hloc_label or any(histogram_label) else 0)
        hloc_descriptor = db_global_descriptors[most_similar_image_idx]
        histogram_vector = np.array(histogram_array).flatten()

        # hloc_descriptor와 histogram_vector를 수평으로 쌓음
        feature_vector = np.hstack((hloc_descriptor, histogram_vector))
        # feature_vector = np.concatenate((hloc_descriptor, np.array(histogram_array).flatten()))
        
        features.append(feature_vector)

    return np.array(features), np.array(labels)


def main():
    dataset = Path('/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur')
    images_query = dataset / 'bag_query/'
    outputs = Path('/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur')

    retrieval_conf = extract_features.confs['netvlad']
    feature_conf = extract_features.confs['superpoint_aachen']
    matcher_conf = match_features.confs['superglue']

    database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/bag-reference-feats-netvlad.h5'
    query_database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/bag-query-feats-netvlad.h5'

    db_data = h5py.File(database_file, 'r')
    query_db_data = h5py.File(query_database_file, 'r')

    db_global_descriptors = np.array([db_data[group]['global_descriptor'][:] for group in db_data.keys() if 'global_descriptor' in db_data[group]])
    query_global_descriptors = np.array([query_db_data[group]['global_descriptor'][:] for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]])
    query_image_names = [group for group in query_db_data.keys() if 'global_descriptor' in query_db_data[group]]

    hm_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference_ver4_edit4_noc.db"
    hm_query_database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query_ver4_edit4_noc.db"

    # 머신러닝 모델 결과에 대한 평가
    X, y = prepare_data(db_data, query_db_data, hm_database_file, hm_query_database_file, query_image_names, db_global_descriptors, query_global_descriptors)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)
    
    model = SVC(C=1, gamma=1, kernel="rbf")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    ml_model_accuracy = accuracy_score(y_test, predictions) * 100

    print("\n")
    print(f"ML 모델 Accuracy: {ml_model_accuracy:.2f}%")
    print("\n")

if __name__ == "__main__":
    main()
