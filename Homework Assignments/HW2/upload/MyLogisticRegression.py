import numpy as np

class MyLogisticRegression:

    def __init__(self, d, max_iters, eta_val):
        self.eta_val = eta_val
        self.d = d
        self.max_iters = max_iters
            

    def fit(self, X, y):
        loss = []
        change_val_list = []
        self.inter = 0
        self.wt = np.random.uniform(-0.01, 0.01, len(X[0])-1)
        for i in range(self.max_iters):
            # print(i)
            self.Xfin = X[:, :2]
            s_WX = 1/(1 + np.exp(-(np.dot(self.wt, self.Xfin.T) + self.inter)))
                   
            
            #randomly draw elements for SGD
            shuffle_X = np.arange(0,len(self.Xfin))
            shuffle_X = shuffle_X.tolist()
            for j in shuffle_X:
                
                err = s_WX[j] - y[j]
                grad_cur = np.dot(self.Xfin[j].T,err)
               
                self.wt = self.wt  - self.eta_val*grad_cur
                     
            loss.append(log_reg_L(y, s_WX, self.d))
            
            
            if i > 0:    
                # An interpretation for the gradient
                
                # print(i-length)
                change_val = (loss[i] - loss[i-1])**2
                # print(grad_mag)
                # print(i)
                change_val_list.append(change_val)
                # print(change_val)
                if change_val < self.d:
                    break
            
        return loss


    def predict(self, X):
        
        Second = 1/(1 + np.exp(-(np.dot(self.wt, X[:, :2].T) + self.inter*X[:, 2])))
        First = 1 - 1/(1 + np.exp(-(np.dot(self.wt, X[:, :2].T) + self.inter*X[:, 2])))

        pblity = np.stack([First, Second], axis=1)
        pre_list = []
        
        for k in range(len(pblity)):
            if pblity[k, 1] >= pblity[k, 0]:
                pre_list.append(1)
            else:
                pre_list.append(0)
                    
        return pre_list


#conditional log-likelihood of logistic regression
def log_reg_L(y, s_WX, d):
    #sigma(-a) = 1-sigma(a) relation is used in the RHS.
    L_p = -(np.matmul(y,(np.log(np.clip(s_WX, d, float("Inf"))))) + np.matmul((1-y),np.log(np.clip(1-s_WX, d, float("Inf")))))
    return L_p

