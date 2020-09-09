import numpy as np
from dataset import Dataset
from matrix import SparseMatrix
import os

class MatrixFactorization():
    def __init__(self, 
                    R, 
                    k, 
                    learning_rate, 
                    learning, 
                    reg_param, 
                    epochs, 
                    model_path,
                    verbose=True):
        """
        :param R: rating matrix
        :param k: latent parameter
        :param learning_rate: alpha on weight update
        :param reg_param: beta on weight update
        :param epochs: training epochs

        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.k = k
        self.learning_rate = learning_rate
        self.learning = learning
        self.reg_param = reg_param
        self.epochs = epochs
        self.model_path = model_path
        self.verbose = verbose

        # Stochastic gradient descent 학습을 위한 샘플 준비
        if self.learning == 'sgd':
            self.sample_row, self.sample_col = self.R.nonzero()
            self.n_samples = len(self.sample_row)

    def extended_train(self):

        """
        학습이 중간에 중단될 경우 이전 에폭의 행렬들을 로딩해서 연장학습 하는 함수

        """

        def load_each_matrix(model_path='./model_b'):
            Q = np.load(os.path.join(model_path,'Q.npy'))
            P = np.load(os.path.join(model_path,'P.npy'))
            Q_b = np.load(os.path.join(model_path,'Q_bias.npy'))
            P_b = np.load(os.path.join(model_path,'P_bias.npy'))
            b = np.load(os.path.join(model_path,'global_bias.npy'))
            return Q, P, Q_b, P_b, b

        self.Q, self.P, self.b_Q, self.b_P, self.b = load_each_matrix()

        if self.learning == 'gd':

            # 학습 시작
            self.training_results = []
            print('training start!')
            for epoch in range(self.epochs):

                # rating이 존재하는 index training
                for i in range(self.num_users):
                    for j in range(self.num_items):
                        if self.R[i, j] > 0:
                            self.gradient_descent(i, j, self.R[i, j])
                cost = self.rmse()
                self.training_results.append((epoch, cost))

                if self.verbose:
                    print("Epoch: %d ; cost = %.4f" % (epoch + 1, cost))

                np.save(os.path.join(self.model_path, 'global_bias'), self.b)
                np.save(os.path.join(self.model_path, 'P_bias'), self.b_P)
                np.save(os.path.join(self.model_path, 'Q_bias'), self.b_Q)
                np.save(os.path.join(self.model_path, 'P'), self.P)
                np.save(os.path.join(self.model_path, 'Q'), self.Q)

            print("Final RMSE = {}".format(self.training_results[self._epochs-1][1]))

        # stochastic gradient descent 학습 시작
        elif self.learning == 'sgd':
            self.partial_train()



    def train(self):
        """
        matrix factorization 학습 함수 (latent weight, bias 업데이트)

        - self.b
          : global bias: input R에서 평가가 매겨진 rating의 평균값을 global bias로 사용

        :return: training_process
        """

        
        # latent 특징 초기화
        self.P = np.random.normal(size=(self.num_users, self.k))
        # self.P = transfer_sparse_matrix(self.num_users, self.k, self.P)

        self.Q = np.random.normal(size=(self.num_items, self.k))
        # self.Q = transfer_sparse_matrix(self.num_items, self.k, self.Q)


        # 각 bias들 초기화
        self.b_P = np.zeros(self.num_users)
        self.b_Q = np.zeros(self.num_items)
        self.b = np.mean(self.R[np.where(self.R != 0)])

        if self.learning == 'gd':

            # 학습 시작
            self.training_results = []
            print('training start!')
            for epoch in range(self.epochs):

                # rating이 존재하는 index training
                for i in range(self.num_users):
                    for j in range(self.num_items):
                        if self.R[i, j] > 0:
                            self.gradient_descent(i, j, self.R[i, j])
                cost = self.rmse()
                self.training_results.append((epoch, cost))

                if self.verbose:
                    print("Epoch: %d ; cost = %.4f" % (epoch + 1, cost))

                np.save(os.path.join(self.model_path, 'global_bias'), self.b)
                np.save(os.path.join(self.model_path, 'P_bias'), self.b_P)
                np.save(os.path.join(self.model_path, 'Q_bias'), self.b_Q)
                np.save(os.path.join(self.model_path, 'P'), self.P)
                np.save(os.path.join(self.model_path, 'Q'), self.Q)

            print("Final RMSE = {}".format(self.training_results[self._epochs-1][1]))

        # stochastic gradient descent 학습 시작
        elif self.learning == 'sgd':
            self.partial_train()


    def partial_train(self):
        """ 
        stochastic gradient descent 학습, 에폭 마다 data random shuffle

        """
        epochs = 0
        print('training start!')
        while epochs <= self.epochs:
            self.training_indices = np.arange(self.n_samples)
            np.random.shuffle(self.training_indices)
            self.stochastic_gradient_descent()
            cost = self.rmse()

            if self.verbose:
                print ("Epoch: %d ; cost = %.4f" % (epoch + 1, cost))

            epochs += 1
            np.save(os.path.join(self.model_path, 'global_bias'), self.b)
            np.save(os.path.join(self.model_path, 'P_bias'), self.b_P)
            np.save(os.path.join(self.model_path, 'Q_bias'), self.b_Q)
            np.save(os.path.join(self.model_path, 'P'), self.P)
            np.save(os.path.join(self.model_path, 'Q'), self.Q)



    def rmse(self):
        """
        root mean square error 계산
        :return: rmse값
        """

        # xi, yi: R[xi, yi]는 nonzero인 value를 의미한다.
        
        xi, yi = self.R.nonzero()
        predicted = self.get_complete_matrix()
        error = 0
        for x, y in zip(xi, yi):
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error/len(xi))


    def gradient(self, error, u, i):
        """
         그래디언트 계산

        :param error: rating - prediction error
        :param u: index of user
        :param i: index of movie
        :return: gradient of latent feature tuple
        """

        dp = (error * self.Q[i, :]) - (self.reg_param * self.P[u, :])
        dq = (error * self.P[u, :]) - (self.reg_param * self.Q[i, :])
        return dp, dq


    def gradient_descent(self, u, i, rating):
        """
        graident descent function

        :param u: user index of matrix
        :param i: movie index of matrix
        :param rating: rating of (user_u, movie_i)
        """

        # get error
        prediction = self.get_prediction(u, i)
        error = rating - prediction

        # update biases
        self.b_P[u] += self.learning_rate * (error - self.reg_param * self.b_P[u])
        self.b_Q[i] += self.learning_rate * (error - self.reg_param * self.b_Q[i])

        # update latent feature
        dp, dq = self.gradient(error, u, i)
        self.P[u, :] += self.learning_rate * dp
        self.Q[i, :] += self.learning_rate * dq

    def stochastic_gradient_descent(self):

        for idx in self.training_indices:
            u = self.sample_row[idx]
            i = self.sample_col[idx]

            # get error
            prediction = self.get_prediction(u, i)
            error = self.R[u,i] - prediction 
            
            # update biases
            self.b_P[u] += self.learning_rate * (error - self.reg_param * self.b_P[u])
            self.b_Q[i] += self.learning_rate * (error - self.reg_param * self.b_Q[i])
            
            # update latent feature
            dp, dq = self.gradient(error, u, i)
            self.P[u, :] += self.learning_rate * dp
            self.Q[i, :] += self.learning_rate * dq


    def get_prediction(self, u, i):
        """
        (user_u, movie_i)의 예측 rating 계산

        :return: prediction of r_ui
        """
        return self.b + self.b_P[u] + self.b_Q[i] + self.P[u, :].dot(self.Q[i, :].T)


    def get_complete_matrix(self):
        """
         PXQ + P_bias + Q_bias + global bias에 대한 matrix를 반환

        - PXQ 행렬에 b_P[:, np.newaxis] , b_Q[np.newaxis:, ]를 통해 각 열에 bias를 더해 줌
        - b는 각 element마다 bias를 더해 줌
        - newaxis: 차원을 추가해 줌 

        :return: complete matrix R^
        """
        return self.b + self.b_P[:, np.newaxis] + self.b_Q[np.newaxis:, ] + self.P.dot(self.Q.T)

    def transfer_sparse_matrix(self, nrow, ncol, data):
        """
        sparse matrix 구조 변환

        :param nrow: length of unique user
        :param ncol: length of unique movie
        :param data: numpy matrix

        :return: SparseMatrix
        """

        sparse = SparseMatrix(nrow, ncol)
        for i in range(len(data)):
            aparse.addRow(i,{k:v for k,v in enumerate(data[i,:])})
        return sparse

