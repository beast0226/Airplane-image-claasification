# Airplane-image-claasification

1. 프로젝트 개요
  - 프로젝트 명 : Airplane image classification
  - 연구 목적 : 낮은 픽셀의 이미지들 중에서 기계학습을 통해 비행기 이미지를 구분함
  
2. 사용한 데이터셋
  - 영상 데이터셋 : 20x20x3 이미지 15,000장
  - 라벨정보 : 2 class 분류 레벨
  
3. 사용한 모델 : Convolutional Neural Network

4. 딥 러닝 라이브러리 : keras

5. 실험 및 결과 요약
학습과정 : 데이터 전처리->모델생성->모델교육->모델평가
이미지에 항공기가 포함되어 있으면 "1", 없으면 "0"으로 파일명에 표시함
학습의 휫 수를 늘릴수록 정확도가 1에 수렴하는 것을 확인함
성능부분은 약 90%정도이며 학습 데이터를 늘림과 알고리즘에 Dropout을 적용함으로써 99%의 정확도를 가져올 수 있음

6. 13번 라인을 수정하면 됨
