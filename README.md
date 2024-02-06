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
실행 파일 **extract_netvlad_features.py**

## STEP 3: Execute Panoptic Histogram
- Extract Query Image Features  
실행 파일 **histogram_query_oneformer.py**  
(Open Dataset) 실행 파일 **histogram_query_oneformer_open.py**  
- Extract Reference Image Features  
실행 파일 **histogram_ref_oneformer.py**
(Open Dataset) 실행 파일 **histogram_ref_oneformer_open.py**

## STETP 4: Image Query
- NetVLAD-based Image Query  
실행 파일 **netvlad_compare_NN_h5.py**
- Panoptic Histogram-based Image Query (Utilizing Machine Learning)  
실행 파일 **histo_ml_svc.py**
- Integrated Algorithm (Utilizing Machine Learning)  
실행 파일 **weight_ml_svc.py**
