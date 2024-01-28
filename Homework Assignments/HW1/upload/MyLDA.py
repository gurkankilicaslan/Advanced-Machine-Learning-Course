import numpy as np

class MyLDA():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val

    def fit(self, X, y):
        
        self.means = []
        
        count = 0
        while count < 2:
            self.means.append(X[y==count].mean(axis=0))
            count += 1
        self.S_w = np.zeros((2,2))
        # print(self.means)
        
        mv = [self.means[0][0], self.means[0][1]]
        
        

        # print(mv)
        for k in range(len(X[y == 0])):
            # row = np.asarray(row)
            row = X[y == 0][k:]
            # print(row.values.tolist())
            # print(row.shape)
            row = row.to_numpy()
            # print(row[0][0])
            row = [row[0][0], row[0][1]]
            # print(row)
            # row.to_numpy()
            # print(row.shape)
            # row = [[row[0,1]],[row[0,1]]]
            # print(row)     
            # print(mv)
            
            for_dot = np.array(row)-np.array(mv)
            empt_arr = np.zeros((2,2))
            for i in range(len(row)):
                for j in range(len(mv)):
                    empt_arr[i][j] = for_dot[i]*for_dot[j]
                    
            self.S_w += empt_arr
        
        # w -> Sw(m_2 - m_1)
        self.w = np.dot(np.linalg.inv(self.S_w), (self.means[1] - self.means[0]))

    def predict(self, X):
        # f(x) = w.T x
        f_x = np.dot(self.w.T, X.T)
        
        pred = []
        for i in range(len(f_x)):
            if f_x[i] >= self.lambda_val:
                pred.append(1)
            else:
                pred.append(0)
        # print(f_x)
        # prediction = np.where(f_x >= self.lambda_val, True, False)
        # print(prediction)
        return pred