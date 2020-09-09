from config import Config
import numpy as np
import collections
import os
config = Config()

class Dataset():
    def __init__(self, 
                data_path, 
                model_path, 
                training_start_timestamp,
                training_end_timestamp,
                test_start_timestamp,
                test_end_timestamp,                
                user_filter_num=50,
                movie_filter_num=50):
        """

        - training_data: raw 학습 데이터
        - test_data: raw 테스트 데이터
        - user_id_dict: {user_id:count}
        - movie_id_dict: {movie_id:count}
        - filtered_user_id, filtered_movie_id: count를 기준으로 데이터 필터링
        - userId2idx, movieId2idx: 각 id에 대한 matrix index

        """
        self.data_path = data_path
        self.model_path = model_path

        self.training_start_timestamp = training_start_timestamp
        self.training_end_timestamp = training_end_timestamp
        self.test_start_timestamp = test_start_timestamp
        self.test_end_timestamp = test_end_timestamp

        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.training_data, self.test_data, self.user_id_dict, self.movie_id_dict = self.split_data(f.readlines())

        self.filtered_user_id, self.filtered_movie_id = self.data_filtering()

        print('training_data_size : ',len(self.training_data))
        print('test_data_size : ',len(self.test_data))

        print('unique user num : ', len(self.user_id_dict))
        print('unique movie num : ', len(self.movie_id_dict))

        print('filtered user size : ',len(self.filtered_user_id))
        print('filtered movie size : ',len(self.filtered_movie_id))


    def get_user_id_dict(self):
        return self.user_id_dict

    def get_movie_id_dict(self):
        return self.movie_id_dict

    def get_training_data(self):
        """
        split된 학습데이터를 반환, 최초 학습시에 movieId2idx와 userId2idx를 저장

        :param data: 학습데이터

        :return: (filtered_user_id * filtered_movie_id) np.array 2d matrix
        """
        self.movieId2idx = get_id2idx(self.filtered_movie_id)
        self.userId2idx = get_id2idx(self.filtered_user_id)

        with open(os.path.join(self.model_path,'userId2idx'), 'w', encoding='utf-8') as f:
            for id in self.userId2idx.keys():
                f.write(str(id) + '\t'+ str(self.userId2idx[id]) + '\n')

        with open(os.path.join(self.model_path,'movieId2idx'), 'w', encoding='utf-8') as f:
            for id in self.movieId2idx.keys():
                f.write(str(id) + '\t'+ str(self.movieId2idx[id]) + '\n')
        return self.training_data

    def get_test_data(self):
        return self.test_data


    def get_matrix_data(self, data):
        """
        학습데이터를 matrix factorization을 위해 matrix로 변경

        :param data: 학습데이터

        :return: (filtered_user_id * filtered_movie_id) np.array 2d matrix
        """
        matrix = []
        for i in range(len(self.filtered_user_id)):
            matrix.append(np.zeros(len(self.filtered_movie_id), dtype=np.float16))
        
        for (user_id, movie_id, rating, _) in data:
            if int(user_id) in self.filtered_user_id and int(movie_id) in self.filtered_movie_id:
                matrix[self.userId2idx[int(user_id)]][self.movieId2idx[int(movie_id)]] = float(rating)

        return np.array(matrix)

    def data_filtering(self, f_num_user=50, f_num_movie=50, flag=False):

        """
        count기준으로 data 필터링

        :param f_num_user: if user count smaller than f_num_user -> remove
        :param f_num_movie: if movie count smaller than f_num_movie -> remove
        :param flag: flag=False -> do not filtering

        :return: filtered each id set
        """

        filtered_movie_id = set()
        filtered_user_id = set()
        
        if flag:
            for id in self.movie_id_dict.keys():
                if self.movie_id_dict[id] > f_num_movie :
                    filtered_movie_id.add(int(id))
            for id in self.user_id_dict.keys():
                if self.user_id_dict[id] > f_num_user :
                    filtered_user_id.add(int(id))
        else:
            for id in self.movie_id_dict.keys():
                if self.movie_id_dict[id]:
                    filtered_movie_id.add(int(id))
            for id in self.user_id_dict.keys():
                if self.user_id_dict[id]:
                    filtered_user_id.add(int(id))

        return filtered_user_id, filtered_movie_id        

    def split_data(self, data: list):

        """
        데이터 파일을 읽은 row list를 stamp기준 학습, 테스트 데이터로 split

        :param data: Entire data
        
        :return: training_data, test_data and userId2count, movieId2count
        """

        training_data, test_data = [], []
        user_id_dict, movie_id_dict = dict(), dict()
        for i in range(1, len(data)):
            user_id, movie_id, rating, timestamp = data[i].split(',')


            # data filtering을 위한 count
            if int(user_id) in user_id_dict.keys():
                user_id_dict[int(user_id)] += 1
            else:
                user_id_dict[int(user_id)] = 1
            
            if int(movie_id) in movie_id_dict.keys():
                movie_id_dict[int(movie_id)] += 1
            else:
                movie_id_dict[int(movie_id)] = 1

            # timestamp 기준으로 training, test data split
            if int(timestamp) >= self.training_start_timestamp and int(timestamp) <= self.training_end_timestamp:
                training_data.append((user_id, movie_id, rating, timestamp))
            elif int(timestamp) >= self.test_start_timestamp and int(timestamp) <= self.test_end_timestamp:
                test_data.append((user_id, movie_id, rating, timestamp))

        return training_data, test_data, user_id_dict, movie_id_dict





def get_id2idx(id_set: set):

    """
    id에 대한 matrix idx

    :param id_set: user or movie id in training data
    
    :return: id2idx of user or movie
    """

    id2idx = {}
    idx = 0
    for id in id_set:
        id2idx[id] = idx
        idx += 1
    return id2idx

    





if __name__ == '__main__':
    dataset = Dataset()
    movie_id_dict = dataset.get_movie_id_dict()
    user_id_dict = dataset.get_user_id_dict()
    writer = open('./data/movie_id','w', encoding='utf8')
    for id, cnt in movie_id_dict.items():
        writer.write(str(id) + '\t' + str(cnt) + '\n')
    writer.close()

    writer = open('./data/user_id','w', encoding='utf8')
    for id, cnt in user_id_dict.items():
        writer.write(str(id) + '\t' + str(cnt) + '\n')

    writer.close()



    


    