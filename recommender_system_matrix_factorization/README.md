## Matrix Factorization for Recommender system
* 개요 : 기존 영화 시청 이력 데이터 기반으로 영화 평점을 예측하는 Matrix Factorization 모델을 구현, 데이터는 [Movielens 20m](https://grouplens.org/datasets/movielens/20m/) 데이터를 활용하였고, 데이터 내의 unique한 user수는 138,493이며, movie수는 26,744로 138,493 * 26,744의 matrix를 만들어서 진행하였다.<br/>
*ref. Matrix Factorization Techniques for Recommender Systems* [paper link](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

# Dependencies
* Python 3.6
* Numpy 1.16.2


# Data (Movielens 20m)
* (explicit feedback인 평점데이터만 활용하였다.)

| dataset | 기간 | timestamp | 데이터 수 |
| --- | --- | --- | --- |
| 학습데이터 | 2005-01-01 ~ 2008-12-31 | 1104505203 ~ 1230735592 | 5,187,587 |
| 평가데이터 | 2009-01-01 ~ 2009-12-31 | 1230735600 ~ 1262271552 | 930,093 |


# Model
> 활용한 모델은 Matrix Factorization(행렬 인수분해) 기본 모델이고, <br/>
> Learning algorithm은 batch size가 1인 Stochastic gradient descent를 사용하였다. <br/>
> (기존에 일반 gradient descent에 비해 학습속도가 대략 8배 정도 빨라짐) <br/>
> 또한 bias term으로 global_bias, user_latent_bias, item_latent_bias 활용하였다.<br/>
> 아래 소스에서 global_bias인 b, user_bias인 b_P, item_bias인 b_Q, 그리고 user_latent인 P, item_latent인 Q가 업데이트 되도록 하였다.
```
def get_prediction(self, u, i):
    return self.b + self.b_P[u] + self.b_Q[i] + self.P[u, :].dot(self.Q[i, :].T)
```


# Usage
* training
```
python3 main.py --mode train --epochs 10 --k 200 --rating_data_path ./data/ml-20m/ratings.csv --saved_model_path ./model --learning_rate 0.002 --reg_param 0.01
```
> (training timestamp & test timestamp의 default값은 위의 학습, 평가 데이터 기준이며, 변경이 필요하다면 <br/>
> --training_start_timestamp, --training_end_timestamp, --test_start_timestamp, --test_end_timestamp <br/>
> 변수 옵션을 통해 수정 가능)


* evaluate
```
python3 main.py --mode eval
```

# Evaluation
| Epochs | 학습 RMSE | 평가 RMSE |
| --- | --- | --- |
| 20 | 0.7471 | 1.262 |

