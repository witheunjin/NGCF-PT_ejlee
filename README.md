# NGCF-PT_ejlee

* NGCF-PT_ejlee(Neural Graph Collaborative Filtering Model with PyTorch)
* Reference: https://github.com/metahexane/ngcf_pytorch_g61
* My Repository URL: https://github.com/witheunjin/NGCF-PT_ejlee

## [DATASETS]
```
|__ path: ~/NGCF-PT_ejlee/data
      |__ /100K/ratings.csv: 100K 개의 Ratings data
      |__ /1M/ratings.csv: 1M 개의 Ratings data
      |__ /20M/ratings.csv: 20M 개의 Ratings data
      |__ /25M/ratings.csv: 25M 개의 Ratings data
      |__ /27M/ratings.csv: 27M 개의 Ratings data
*20M이상의 Dataset을 Training하려고 하면 'out of memory' 문제발생
```

## [HOW TO USE(EXECUTE)]
~/NGCF-PT_ejlee에서 다음과 같은 명령어를 사용하여 실행
* `$ python run_ngcf.py` (1M의 Dataset에 대한 Training 결과(400개의 Epochs) 출력)
* `$ python run_ngcf.py --data_size 100K` (100K dataset에 대한 Training 결과(400개의 Epochs) 출력)
* `$ python run_ngcf.py --data_size 1M --n_epochs 100` (1M Dataset에 대한 Training 결과(100개의 Epochs) 출력)

## [RESULTS]
* NGCF-PT_ejlee_100K_result: 100K Dataset에 대한 Training 결과(Epoch 100)
* NGCF-PT_ejlee_1M_result: 1M Dataset에 대한 Training 결과(Epoch 400)
