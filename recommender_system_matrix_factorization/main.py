import numpy as np
import utils
from model import MatrixFactorization
from dataset import Dataset
import argparse
import os

def eval(data, model_path):
    
    """
    학습이 완료된 matrix들을 loading하여 test_data에 대한 rmse 평가
    :param data: list of test data
    
    :return: rmse
    """

    pred_ratings, true_ratings = [],[]
    Q, P, Q_b, P_b, b = utils.load_each_matrix(model_path)
    userId2idx, movieId2idx = utils.load_id2idx(model_path)
    complete_matrix = b + P_b[:, np.newaxis] + Q_b[np.newaxis:, ] + P.dot(Q.T)

    with open(os.path.join(model_path,'result.csv'), 'w', encoding='utf8') as f:
        for (user_id, movie_id, rating, timestamp) in data:
            true_ratings.append(rating)
            pred = complete_matrix[int(userId2idx[user_id]), int(movieId2idx[movie_id])]
            if pred < 0 or pred > 8:
                pred = b
            pred_ratings.append(pred)
            f.write(str(user_id) + ',' + str(movie_id) + ',' + str(pred) + ',' + str(timestamp))
            
    return utils.rmse(pred_ratings,true_ratings)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train', choices=['train','eval'])
    args.add_argument('--epochs', type=int, default=15)
    args.add_argument('--k', type=int, default=200) # number of latent feature
    args.add_argument('--rating_data_path', type=str, default='./data/ml-20m/ratings.csv')
    args.add_argument('--saved_model_path', type=str, default='./model')
    args.add_argument('--learning_rate', type=float, default=0.002)
    args.add_argument('--learning', type=str, default='sgd', choices=['gd','sgd'])
    args.add_argument('--reg_param', type=float, default=0.01)
    args.add_argument('--training_start_timestamp', type=int, default=1104505203)
    args.add_argument('--training_end_timestamp', type=int, default=1230735592)
    args.add_argument('--test_start_timestamp', type=int, default=1230735600)
    args.add_argument('--test_end_timestamp', type=int, default=1262271552)


    config = args.parse_args()

    if config.mode == 'train':
        dataset = Dataset(data_path = config.rating_data_path,
                          model_path = config.saved_model_path,
                          training_start_timestamp = config.training_start_timestamp,
                          training_end_timestamp = config.training_end_timestamp,
                          test_start_timestamp = config.test_start_timestamp,
                          test_end_timestamp = config.test_end_timestamp)
        training_matrix = dataset.get_matrix_data(dataset.get_training_data())
        print('make training matrix complete!')
        test_data = dataset.get_test_data()
        factorizer = MatrixFactorization(R=training_matrix, 
                                         k=config.k, 
                                         learning_rate=config.learning_rate, 
                                         learning=config.learning,
                                         reg_param=config.reg_param, 
                                         epochs=config.epochs, 
                                         model_path=config.saved_model_path ,
                                         verbose=True)
        factorizer.train()

    elif config.mode == 'eval':
        dataset = Dataset(data_path = config.rating_data_path,
                          model_path = config.saved_model_path,
                          training_start_timestamp = config.training_start_timestamp,
                          training_end_timestamp = config.training_end_timestamp,
                          test_start_timestamp = config.test_start_timestamp,
                          test_end_timestamp = config.test_end_timestamp)

        test_data = dataset.get_test_data()
        print('test rmse = {}'.format(eval(test_data, config.saved_model_path)))
