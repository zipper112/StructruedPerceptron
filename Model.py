import numpy as np

training_path = r'D:\Python\Lib\site-packages\pyhanlp\static\data\test\icwb2-data\training\msr_training.utf8'
testing_path = r'D:\Python\Lib\site-packages\pyhanlp\static\data\test\icwb2-data\testing\msr_test.utf8'
voc_path = r'./voc.txt'

class StructuredPerception():
    def __init__(self, vocpath='./voc.txt', lr=1, modelpath=None):
        if not modelpath:
            self.__initParameters(vocpath, lr)
            self.__initMatrix()
        else:
            self.load(modelpath, vocpath)

    def __initParameters(self, vocpath='./voc.txt', lr=0.01):
        with open(vocpath, encoding='utf-8') as rs:
            self.idx2word = rs.read().split()
        
        self.word2idx = {word: i for i, word in enumerate(self.idx2word)}
        self.idx2BMSE = ['B', 'S', 'M', 'E']
        self.BMSE2idx = {tag: i for i, tag in enumerate(self.idx2BMSE)}
        self.hid_size = 4
        self.obs_size = len(self.word2idx)
        self.feature_size = self.hid_size ** 2 + self.hid_size + self.hid_size * self.obs_size
        self.lr = lr

    def __initMatrix(self, weight=None):
        if not isinstance(weight, np.ndarray):
            self.weight = np.zeros(shape=(self.feature_size), dtype=np.float32)
        else:
            self.weight = weight
        self.start_vector = self.weight[: self.hid_size]
        self.transition_matrix = self.weight[self.hid_size : self.hid_size ** 2 + self.hid_size].reshape(self.hid_size, -1)
        self.emission_matrix = self.weight[self.hid_size ** 2 + self.hid_size:].reshape(self.hid_size, -1)     

    def fit(self, data, iter=2):
        for eopch in range(iter):
            for i, single in enumerate(data):
                E = self.emission_matrix
                T = self.transition_matrix
                S = self.start_vector

                y = self.getFeatureVector(single)
                tmp = ''.join(single.split())
                y_hat = self.getFeatureVector(' '.join(self.__Segment(tmp)))
                self.weight += self.lr * (y - y_hat)

                if i % 200 == 0:
                    print(self.__Segment('我爱中国', S, T, E))
                    print(self.__Segment('塞尔维亚最强大', S, T, E))
                    print(self.__Segment('全世界的无产者们联合起来', S, T, E))
                    print('<---------EPOCH:{}--> {}%-------->'.format(eopch + 1, round(i * 100 / len(data), 2)))

    def vterbi_decode(self, x):
        theta = self.start_vector + self.emission_matrix[:, x[0]].T
        path = np.zeros(shape=(len(x), self.hid_size), dtype=np.int32)

        for t in range(1, len(x)):
            tmp_theta = np.zeros(shape=(self.hid_size), dtype=np.int32)
            for i in range(self.hid_size):
                s = theta + self.transition_matrix[:, i].T
                tmp_theta[i] = np.max(s) + self.emission_matrix[i, x[t]]
                path[t][i] = np.argmax(s)
            theta = tmp_theta

        res = []
        max_pob_pos = np.argmax(theta)

        for t in range(len(x) - 1, -1, -1):
            res.append(max_pob_pos)
            max_pob_pos = path[t][max_pob_pos]

        return res[-1::-1]

    def getFeatureVector(self, x: str):
        BMSE, utf8 = [], []
        
        for chr in x.strip():
            if chr != ' ':
                utf8.append(self.word2idx[chr])
        x = x.strip().split()

        for word in x:
            if len(word) == 1:
                BMSE.append('S')
            else:
                BMSE.append('B')
                BMSE += ['M'] * (len(word) - 2)
                BMSE.append('E')
        BMSE = [self.BMSE2idx[i] for i in BMSE]
        
        transition_matrix, emission_matrix = np.zeros(shape=(4, 4), dtype=np.int32), np.zeros(shape=(4, self.obs_size), dtype=np.int32)
        start_vector = np.zeros(shape=(4), dtype=np.int32)
        start_vector[BMSE[0]] = 1
        
        for i in range(len(BMSE)):
            if i >= 1:
                transition_matrix[BMSE[i - 1]][BMSE[i]] += 1
            emission_matrix[BMSE[i]][utf8[i]] += 1
        return np.concatenate([start_vector, transition_matrix.reshape(-1), emission_matrix.reshape(-1)], axis=0)

    def Segment(self, x):
        BMSE = self.vterbi_decode(self.toIdx(x))
        ans, tmp = [], ''
        for i in range(len(x)):
            if tmp and (BMSE[i] == 0 or BMSE[i] == 1):
                ans.append(tmp)
                tmp = x[i]
            else:
                tmp += x[i]
        if tmp:
            ans.append(tmp)
        return ans

    def toIdx(self, x):
        return [self.word2idx[chr] for chr in x]
    
    def save(self, path):
        np.save(path, self.weight)

    def load(self, path, voc='./voc.txt'):
        self.weight = np.load(path)
        self.__initParameters(voc)
        self.__initMatrix(self.weight)

train_data = None
with open(training_path, encoding='utf-8', mode='r') as rs:
    train_data = list(filter(lambda x: len(x), [sentence.strip() for sentence in rs.read().split('\n')]))

np.random.shuffle(train_data)

model = StructuredPerception(lr=1e-3)
model.fit(train_data, iter=2)
print(model.transition_matrix)
print(model.Segment('我爱中国'))
print(model.Segment('塞尔维亚最强大'))
print(model.Segment('全世界的无产者联合起来'))
print(model.Segment('我爱北京天安门'))
model.save('./model.npy')


# model = StructuredPerception(modelpath='./model.npy')
# print(model.Segment('另外旅游、侨汇也是经济收入的重要组成部分，制造业规模相对较小。'))
