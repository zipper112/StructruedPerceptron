import numpy as np
import re

class Weight:
    def __init__(self):
        self.values = dict() # 所有feature的存储，包括转移的状态
    
    def update_weight(self, key, delta):
        if key not in self.values:
            self.values[key] = 0

        self.values[key] += delta
    
    def get_value(self, key, default=0):
        if key not in self.values:
            return default
        else:
            return self.values[key]
    

class StructuredPerceptron:
    def __init__(self):
        self.weight = Weight()
    
    def extract_features(self,x):
        for i in range(len(x)):
            left2=x[i-2] if i-2 >=0 else '#'
            left1=x[i-1] if i-1 >=0 else '#'
            mid=x[i]
            right1=x[i+1] if i + 1 < len(x) else '#'
            right2=x[i+2] if i + 2 < len(x) else '#'
            features=['1' + mid,'2' + left1,'3' + right1,
                    '4'+left2+left1,'5'+left1+mid,'6'+mid+right1,'7'+right1+right2]
            yield features
    
    def veterbi_decode(self, x):
        transition_matrix = np.array([[self.weight.get_value(str(i) + ':' + str(j)) for j in range(4)] 
                                                                                    for i in range(4)])
        emisstion_matrix = np.array([[sum(self.weight.get_value(str(tag) + feature)
                                                for feature in features) for tag in range(4)]
                                                for features in self.extract_features(x)])

        path = []
        alpha = emisstion_matrix[0]

        for i in range(len(x) - 1):
            alpha = alpha.reshape(-1, 1) + transition_matrix + emisstion_matrix[i + 1]
            path.append(list(np.argmax(alpha, axis=0)))
            alpha = np.max(alpha, axis=0)
        
        res = [np.argmax(alpha)]
        idx = res[0]

        for p in reversed(path):
            idx = p[idx]
            res.append(idx)

        return list(reversed(res))       
    
    def _tagging(self, sentence: str):
        """
        (B, M, E, S)
        """
        sentence = sentence.strip().split()
        tagseq = []
        for word in sentence:
            if len(word) == 1:
                tagseq.append(3)
            else:
                tagseq.extend([0] + [1] * (len(word) - 2)+ [2])
        return ''.join(sentence), tagseq

    def train(self, corpus: str, iter=200, encoding='utf-8'):
        corpus = list(open(corpus, encoding=encoding))
        report_num = len(corpus) // 50

        for epoch in range(iter):
            for j, sentence in enumerate(corpus):
                if len(sentence.strip()) == 0:
                    continue
                x, y = self._tagging(sentence)
                y_hat = self.veterbi_decode(x)
                if y_hat != y:
                    self.update(x, y, 1)
                    self.update(x, y_hat, -1)
 
                if j % report_num == 0:
                    print('ep:{} ---- {}%'.format(epoch, round(j / len(corpus) * 100, 2)))
                    print(self.segment('我爱中国'))
                    print(self.segment('塞尔维亚最强大'))
                    print(self.segment('全世界的无产者联合起来！'))

            print(self.score('msr_test.utf8', 'msr_test_gold.utf8'))
    
    def update(self, x, y, delta):
        for i, features in zip(range(len(y)), self.extract_features(x)):
            for feature in features:
                self.weight.update_weight(str(y[i]) + feature, delta)
        for i in range(1, len(x)):
            self.weight.update_weight(str(y[i - 1]) + ':' + str(y[i]), delta)
    
    def segment(self, x):
        y = self.veterbi_decode(x)
        res, tmp = [], ''
        
        for i in range(len(x)):
            if tmp and (y[i] == 0 or y[i] == 3):
                res.append(tmp)
                tmp = ''
            tmp += x[i]
        if tmp: res.append(tmp)
        return res
    
    def toRegion(self, x: str):
        res = []
        st = 1
        for word in x.strip().split():
            res.append((st, st + len(word) - 1))
            st = len(word) + st
        return res
    
    def score(self, test, target):
        A, B, ANB = 0, 0, 0
        with open(test, encoding='utf-8', mode='r') as ts, open(target, encoding='utf-8', mode='r') as tg:
            for a, b in zip(ts, tg):
                tmpstr = re.sub(r'[ \n  ]', '', a).strip()
                pre = self.segment(tmpstr)
                tmpa, tmpb = set(self.toRegion(' '.join(pre))), set(self.toRegion(b.strip()))
                A += len(tmpa)
                B += len(tmpb)
                ANB += len(tmpa & tmpb)
        p, r = ANB / B, ANB / A
        return p, r, 2 * p * r / (p + r)

model = StructuredPerceptron()
model.train('msr_training.utf8')
# print(model.score('aaa.txt', 'mini.txt'))