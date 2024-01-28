import numpy as np

class MySVM:

    def __init__(self, d, max_iters, eta_val, c):
        self.eta_val = eta_val
        self.d = d
        self.max_iters = max_iters
        self.c = c
        
    def fit(self, X, y):
        self.inter = 0
        self.wt = np.random.uniform(-0.01, 0.01, len(X[0])-1)
        hinge_loss = []
        
        loss = []
        change_val_list = []
        
        
        for i in range(self.max_iters):
            # print(self.wt.shape)
            
            Xfin = X[:, :2] # Only features
            Xin = X[:, 2] # Only inter
            
            hin_loss = hlComp(y,self.wt,self.inter,Xfin,Xin)
            
            if 1 - hin_loss > 0:
                hinge_loss.append(1 - hin_loss)
            else:
                hinge_loss.append(0)
                
            #randomly draw elements for SGD
            shuffle_X = np.arange(0,len(Xfin))
            shuffle_X = shuffle_X.tolist()
            for j in shuffle_X:
                # print(j)
                
                err = np.matmul(Xfin[j],self.wt) - y[j]
                
                grad_cur = np.dot(Xfin[j].T, err)
                
                
                indx = len(hinge_loss) - 1
                self.wt = self.wt - self.eta_val*grad_cur
                loss.append(SVM_obj(self.wt, self.c, hinge_loss[indx]))
                
                    
            if i > 0:    
                # An interpretation for the gradient
                
                # print(i-length)
                change_val = (loss[i] - loss[i-1])**2
                # print(i)
                change_val_list.append(change_val)
                # print(change_val)
                if change_val < self.d:
                    break
                
        return loss
            

    def predict(self, X):

        prediction = np.matmul(X[:, :2],self.wt) + self.inter*X[:, -1]
        
        pre_list = []
        for i in prediction:
            if i < 0:
                lbl = -1
            else:
                lbl = 1
            pre_list.append(lbl)
            
        return pre_list

def hlComp(y,wt,inter,Xfin,Xin):
    hinlos = np.matmul(y,(np.matmul(Xfin,wt) + inter*Xin))
    return hinlos
    
def SVM_obj(wt, c, hinge_loss):
    L_p = -((1/2)*(np.linalg.norm(wt, 2)) + c*np.sum([hinge_loss,-0.2]))
    return L_p

