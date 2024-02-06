# Hierarchical-Localization

## STEP 1: Image Preprocessing  
- Rename File  
  - 실행 파일 **image_rename.py**
``` shell
$ python image_rename.py --input {images folder path} --output {images folder path}
```
(cw input folder = /home/ubuntu/cw/data/nuScenes/CAM_BACK_SPLIT)  
(cw output folder = /home/ubuntu/cw/data/nuScenes/rename)   

- Random Split for Query/Reference Images  
  - 실행 파일 **image_split_random.py**
``` shell
$ python image_split_random.py --input {source images folder path} --query {query images folder path} --ref {ref images folder path}
```
(cw input folder = "/home/ubuntu/cw/bag_to_image")  
(cw query output folder = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/bag_query")  
(cw reference output folder = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/bag_query")  

## STEP 2: Execute NetVLAD
- Extract Image Features  
  - 실행 파일 **extract_netvlad_features.py**

## STEP 3: Execute Panoptic Histogram
- Extract Query Image Features  
  - 실행 파일 **histogram_query_oneformer.py**  
  - (Open Dataset) 실행 파일 **histogram_query_oneformer_open.py**
``` shell
$ python histogram_query_oneformer.py --image {images folder path} --output {database file path}
```
(cw image_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/bag_query")  
(cw database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query.db")


- Extract Reference Image Features  
  - 실행 파일 **histogram_ref_oneformer.py**
  - (Open Dataset) 실행 파일 **histogram_ref_oneformer_open.py**
``` shell
$ python histogram_ref_oneformer.py --image {images folder path} --output {database file path}
```
(cw image_directory = "/home/ubuntu/cw/Hierarchical-Localization/datasets/sacre_coeur/bag_reference")  
(cw database_file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference.db")

## STETP 4: Image Query
- NetVLAD-based Image Query  
  - 실행 파일 **netvlad_compare_NN_h5.py**
``` shell
$ python netvlad_compare_NN_h5.py --query {query database file path} --ref {ref database file path}
```
(cw query database file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/CAM-BACK-1000-global-feats-netvlad.h5')  
(cw ref database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/CAM-BACK-1000-query-feats-netvlad.h5')  

- Panoptic Histogram-based Image Query (Utilizing Machine Learning)  
  - 실행 파일 **histo_ml_svc.py**
``` shell
$ python histo_ml_svc.py --query {query database file path} --ref {ref database file path}
```

- Integrated Algorithm (Utilizing Machine Learning)  
  - 실행 파일 **weight_ml_svc.py**  
(cw NetVLAD query database file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/CAM-BACK-1000-global-feats-netvlad.h5')   
(cw NetVLAD ref database_file = '/home/ubuntu/cw/Hierarchical-Localization/outputs/sacre_coeur/NetVlad/CAM-BACK-1000-query-feats-netvlad.h5')   
(cw Histogram query database file = "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_query.db")   
(cw Histogram ref database file =  "/home/ubuntu/cw/Hierarchical-Localization/datasets/outputs/DB/bag_reference.db")
  
