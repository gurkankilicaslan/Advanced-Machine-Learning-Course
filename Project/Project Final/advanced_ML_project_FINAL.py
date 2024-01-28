# -*- coding: utf-8 -*-
"""
Created on Tue Sept 21 18:37:55 2023

@author: grknk
"""

import h5py
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn


def HumActClass():
    class SetNetwork():
    
        def __init__(self,big,num):
            
            self.num= num
            self.big= big
            self.laybig= len(big) -1
            self.percbig= None
            self.percpar = None
            self.flaypar=None
            self.percmom = None
            self.flaymom= None
            self.initializeLearning()
            
        def initializeLearning(self):
            num= self.num
            big= self.big
            laybig= self.laybig
            
            weights=[]
            bias= []
            
            for i in range(1,laybig):
                weights.append(np.random.uniform(-(np.sqrt(6)/np.sqrt(big[1] + big[i+1])),(np.sqrt(6)/np.sqrt(big[1] + big[i+1])),size=(big[i],big[i+1])))
                bias.append(np.zeros((1,big[i+1])))
                
            self.percbig= len(weights)
            params= {"weights":weights,"bias":bias}
            mom= {"weights": [0]*self.percbig,"bias": [0]*self.percbig}
            self.percpar= params
            self.percmom= mom
            
            ne= big[0]
            he= big[1]
            ze= ne + he
            
            if num== 1:
                weightsih= np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)),(np.sqrt(6)/np.sqrt(ne+he)),size= (ne,he))
                weightshh= np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)),(np.sqrt(6)/np.sqrt(he+he)),size= (he,he))
                bias= np.zeros((1,he))
                
                params= {"weightsih": weightsih,"weightshh": weightshh,"bias": bias}
                
            if num== 2:
                weightsf= np.random.uniform(-(np.sqrt(6)/np.sqrt(ze+he)),(np.sqrt(6)/np.sqrt(ze+he)),size= (ze,he))
                biasf= np.zeros((1,he))
                weightsi= np.random.uniform(-(np.sqrt(6)/np.sqrt(ze+he)),(np.sqrt(6)/np.sqrt(ze+he)),size= (ze,he))
                biasi= np.zeros((1,he))
                weightsc= np.random.uniform(-(np.sqrt(6)/np.sqrt(ze+he)),(np.sqrt(6)/np.sqrt(ze+he)),size= (ze,he))
                biasc= np.zeros((1,he))
                weightso= np.random.uniform(-(np.sqrt(6)/np.sqrt(ze+he)),(np.sqrt(6)/np.sqrt(ze+he)),size= (ze,he))
                biaso= np.zeros((1,he))
               
                params= {"weightsf": weightsf,"biasf": biasf,"weightsi": weightsi,"biasi": biasi,"weightsc": weightsc,"biasc": biasc,"weightso": weightso,"biaso": biaso}
                
            if num== 3:
                weightsz= np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)),(np.sqrt(6)/np.sqrt(ne+he)),size=(ne,he))
                uzaz= np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)),(np.sqrt(6)/np.sqrt(he+he)),size=(he,he))
                biasz= np.zeros((1,he))
                weightsr= np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)),(np.sqrt(6)/np.sqrt(ne+he)),size=(ne,he))
                uzar= np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)),(np.sqrt(6)/np.sqrt(he+he)),size=(he,he))
                biasr= np.zeros((1,he))
                weightsh= np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)),(np.sqrt(6)/np.sqrt(ne+he)),size=(ne,he))
                uzah= np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)),(np.sqrt(6)/np.sqrt(he+he)),size=(he,he))
                biash= np.zeros((1,he))
                
                params= {"weightsz":weightsz,"uzaz": uzaz,"biasz": biasz,"weightsr":weightsr,"uzar": uzar,"biasr": biasr,"weightsh":weightsh,"uzah": uzah,"biash": biash}
                
            mom= dict.fromkeys(params.keys(),0)
            self.flaypar= params
            self.flaymom= mom
            
        def modify(self,lrate,momc,gradflay,gradperc):
            
            flaypar= self.flaypar
            flaymom= self.flaymom
            percpar= self.percpar
            percmom= self.percmom
            
            for k in self.flaypar:
                flaymom[k]= lrate * gradflay[k] + momc * flaymom[k]
                flaypar[k]= flaypar[k] -(lrate * gradflay[k] + momc * flaymom[k])
                
            for i in range(self.percbig):
    
                percmom["weights"][i]= lrate*gradperc["weights"][i] + momc * percmom["weights"][i]
                percmom["bias"][i]= lrate * gradperc["bias"][i] + momc * percmom["bias"][i]
                
                percpar["weights"][i]= percpar["weights"][i] -(lrate*gradperc["weights"][i] + momc * percmom["weights"][i])
                percpar["bias"][i]= percpar["bias"][i] -(lrate * gradperc["bias"][i] + momc * percmom["bias"][i])
            
            self.flaypar= flaypar
            self.flaymom= flaymom
            self.percpar= percpar
            self.percmom= percmom
            
        def forwardperc(self,a,weights,bias,b):
            return self.activ((a @ weights + bias),b)
        
        def backwardperc(self,weights,o,der,chan):
            
            dweights= o.T @ chan
            dbias= chan.sum(axis=0,keepdims=True)
            chan= der * (chan @ weights.T)
            return dweights,dbias,chan 
        
        def workout(self,a,b,lrate,momc,batch,epoch):
            traininglossl,validationlossl,trainingaccuracyl,validationaccuracyl= [],[],[],[]
            
            valbig= int(a.shape[0]/10)
            po= np.random.permutation(a.shape[0])
            vala= a[po][:valbig]
            valb= b[po][:valbig]
            a= a[po][valbig:]
            b= b[po][valbig:]
            it= int(a.shape[0]/batch)
            
            for i in range(epoch):           
                go= 0
                stop= batch
                po= np.random.permutation(a.shape[0])
                a= a[po]
                b= b[po]
                
                for j in range(it):
                    gues,o,der,h,hder,co= self.forwardgo(a[go:stop])
                    
                    chan= gues
                    chan[b[go:stop]==1]= chan[b[go:stop]==1]-1
                    chan= chan/batch
                    
                    gradflay,gradperc= self.backwardgo(a[go:stop],o,der,chan,h,hder,co)
                    
                    self.modify(lrate,momc,gradflay,gradperc)
                    
                    go= stop
                    stop= stop + batch
    
                trainingaccuracy,trainingloss= calcul(a,b,self.guess,self.CE)                
                validationaccuracy,validationloss= calcul(vala,valb,self.guess,self.CE)
                    
                print("Epoch: %d=== Training Loss: %.3f,Validation Loss: %.3f,Training Accuracy: %.3f,Validation Accuracy: %.3f"% (i+1,trainingloss,validationloss,trainingaccuracy,validationaccuracy))
                traininglossl.append(trainingloss)
                validationlossl.append(validationloss)
                trainingaccuracyl.append(trainingaccuracy)
                validationaccuracyl.append(validationaccuracy)
                
                
                if i>15:
                    convergence= sum(validationlossl[-16:-1]) / len(validationlossl[-16:-1])
                    if (convergence -0.001) < validationloss < (convergence + 0.001):
                        print("\nTraining stopped since validation C-E reached convergence.")
                        return {"traininglossl": traininglossl,"validationlossl": validationlossl,"trainingaccuracyl": trainingaccuracyl,"validationaccuracyl": validationaccuracyl}
            return {"traininglossl": traininglossl,"validationlossl": validationlossl,"trainingaccuracyl": trainingaccuracyl,"validationaccuracyl": validationaccuracyl}
        
        def activ(self,a,A):
            if A=="softmax":
                activ= np.exp(a) / np.sum(np.exp(a),axis=1,keepdims=True)
                deriv= None
                return activ,deriv
            
            if A== "tanh":
                activ= np.tanh(a)
                deriv= 1 -activ**2
                return activ,deriv
            
            if A== "relu":
                activ= a * (a>0)
                deriv= 1 * (a>0)
                return activ,deriv
            
            if A== "sigmoid":
                
                activ= 1/(1 + np.exp(-a))
                # print(activ)
                deriv= activ * (1-activ)
                return activ,deriv
        
        def forwardrnn(self,a,flaypar):
            
            ne,te,de= a.shape
            
            weightsih= flaypar["weightsih"]
            weightshh= flaypar["weightshh"]
            bias= flaypar["bias"]
            
            hbefore= np.zeros((ne,self.big[1]),dtype=np.float16)
            h= np.empty((ne,te,self.big[1]),dtype=np.float16)
            hder = np.empty((ne,te,self.big[1]),dtype=np.float16)
            
            for k in range(te):
                h[:,k,:],hder[:,k,:]= self.activ((a[:,k,:] @ weightsih + hbefore @ weightshh + bias),"tanh")
                hbefore= h[:,k,:]
                
            return h,hder
        
        def backwardrnn(self,a,h,hder,chan,flaypar):
            
            ne,te,de= a.shape            
            weightshh= flaypar["weightshh"]            
            dweightsih,dweightshh,dbias= 0,0,0
            
            for k in reversed(range(te)):
                hbefore= ifing1(h,self.big[1],ne,k)
                hbeforeder= ifing2(hder,k)                
                dweightsih= dweightsih + a[:,k,:].T @ chan
                dweightshh= dweightshh + hbefore.T @ chan
                dbias= dbias + chan.sum(axis=0,keepdims=True)
                chan= hbeforeder * (chan@weightshh)
                
            return {"weightsih": dweightsih,"weightshh": dweightshh,"bias": dbias}
        
        def forwardgo(self,a):
            
            num= self.num
            percpar= self.percpar
            flaypar= self.flaypar
            o,der= [],[]            
            h,hder,co= 0,0,0
            
            if num== 1:
                h,hder= self.forwardrnn(a,flaypar)
                o.append(h[:,-1,:])
                der.append(hder[:,-1,:])
            
            if num== 2:
                h,co= self.forwardlstm(a,flaypar)
                o.append(h)
                der.append(1)
                
            if num== 3:
                h,co= self.forwardgru(a,flaypar)
                o.append(h)
                der.append(1)
                
            for i in range(self.percbig-1):
                activ,deriv= self.forwardperc(o[-1],percpar["weights"][i],percpar["bias"][i],"relu")
                o.append(activ)
                der.append(deriv)
            
            gues= self.forwardperc(o[-1],percpar["weights"][-1],percpar["bias"][-1],"softmax")[0]
            
            return gues,o,der,h,hder,co
        
        def backwardgo(self,a,o,der,chan,h,hder,co):
            
            num= self.num
            percpar= self.percpar
            flaypar= self.flaypar
            
            gradflay= dict.fromkeys(percpar.keys())
            gradperc= {"weights": [0] * self.percbig,"bias": [0]*self.percbig}
            
            for i in reversed(range(self.percbig)):
                gradperc["weights"][i],gradperc["bias"][i],chan= self.backwardperc(percpar["weights"][i],o[i],der[i],chan)
                
            if num== 1:
                gradflay= self.backwardrnn(a,h,hder,chan,flaypar)
            if num== 2:
                gradflay= self.backwardlstm(co,flaypar,chan)
            if num== 3:
                gradflay= self.backwardgru(a,co,flaypar,chan)
                
            return gradflay,gradperc
                    
        def forwardlstm(self,a,flaypar):
            
            ne,te,de= a.shape
            
            weightsi,biasi= flaypar["weightsi"],flaypar["biasi"]
            weightsf,biasf= flaypar["weightsf"],flaypar["biasf"]
            weightso,biaso= flaypar["weightso"],flaypar["biaso"]
            weightsc,biasc= flaypar["weightsc"],flaypar["biasc"]
            
            hbefore,cbefore= np.zeros((ne,self.big[1])),np.zeros((ne,self.big[1]))
            zi= np.empty((ne,te,de + self.big[1]))
            hfi= 0
            
            # print(self.big[1])
            hii= np.empty((ne,te,self.big[1]),dtype=np.float16)
            hci= np.empty((ne,te,self.big[1]),dtype=np.float16)
            hoi= np.empty((ne,te,self.big[1]),dtype=np.float16)
            tanhci= np.empty((ne,te,self.big[1]),dtype=np.float16)
            ci= np.empty((ne,te,self.big[1]),dtype=np.float16)
            tanhcdi= np.empty((ne,te,self.big[1]),dtype=np.float16)
            hfdi= np.empty((ne,te,self.big[1]),dtype=np.float16)
            hidi= np.empty((ne,te,self.big[1]),dtype=np.float16)
            hcdi= np.empty((ne,te,self.big[1]),dtype=np.float16)
            hodi= np.empty((ne,te,self.big[1]),dtype=np.float16)
            
            for k in range(te):
                zi[:,k,:]= np.column_stack((hbefore,a[:,k,:]))
                
                hfi,hfdi[:,k,:]= self.activ(zi[:,k,:] @ weightsf + biasf,"sigmoid")
                hii[:,k,:],hidi[:,k,:]= self.activ(zi[:,k,:] @ weightsi + biasi,"sigmoid")
                hci[:,k,:],hcdi[:,k,:]= self.activ(zi[:,k,:] @ weightsc + biasc,"tanh")
                hoi[:,k,:],hodi[:,k,:]= self.activ(zi[:,k,:] @ weightso + biaso,"sigmoid")
                
                ci[:,k,:]= hfi * cbefore + hii[:,k,:] * hci[:,k,:]
                tanhci[:,k,:],tanhcdi[:,k,:]= self.activ(ci[:,k,:],"tanh")
                hbefore= hoi[:,k,:] * tanhci[:,k,:]
                cbefore= ci[:,k,:]
                
                co= {"zi": zi,"ci": ci,"tanhci": (tanhci,tanhcdi),"hfdi": hfdi,"hii": (hii,hidi),"hci": (hci,hcdi),"hoi": (hoi,hodi)}
                
            return hbefore,co
        
        def backwardlstm(self,co,flaypar,chan):
            
            weightsf= flaypar["weightsf"]
            weightsi= flaypar["weightsi"]
            weightsc= flaypar["weightsc"]
            weightso= flaypar["weightso"]
            
            zi= co["zi"]
            ci= co["ci"]
            tanhci,tanhcdi= co["tanhci"]
            hfdi= co["hfdi"]
            hii,hidi= co["hii"]
            hci,hcdi= co["hci"]
            hoi,hodi= co["hoi"]
            te= zi.shape[1]
            
            dweightsf,dweightsi,dweightsc,dweightso,dbiasf,dbiasi,dbiasc,dbiaso= 0,0,0,0,0,0,0,0
            
            for k in reversed(range(te)):
                cbefore= ifing2(ci,k)                    
                    
                dci= chan * hoi[:,k,:] * tanhcdi[:,k,:]
                dhfi= dci * cbefore * hfdi[:,k,:]
                dhii= dci * hci[:,k,:] * hidi[:,k,:]
                dhci= dci * hii[:,k,:] * hcdi[:,k,:]
                dhoi= chan * tanhci[:,k,:] * hodi[:,k,:]
                
                dweightsf,dbiasf= lstmproc(dweightsf,dbiasf,zi[:,k,:],dhfi)
                dweightsi,dbiasi= lstmproc(dweightsi,dbiasi,zi[:,k,:],dhii)
                dweightsc,dbiasc= lstmproc(dweightsc,dbiasc,zi[:,k,:],dhci)
                dweightso,dbiaso= lstmproc(dweightso,dbiaso,zi[:,k,:],dhoi)
                
                df= lstmdu(dhfi,weightsf,self.big[1])
                di= lstmdu(dhii,weightsi,self.big[1])
                dc= lstmdu(dhci,weightsc,self.big[1])
                do= lstmdu(dhoi,weightso,self.big[1])
                
                
                chan= (df + di + dc + do)
                
            return {"weightsf": dweightsf,"biasf": dbiasf,"weightsi": dweightsi,"biasi": dbiasi,"weightsc": dweightsc,"biasc": dbiasc,"weightso": dweightso,"biaso": dbiaso}
        
        def forwardgru(self,a,flaypar):
            weightsz= flaypar["weightsz"]
            weightsr= flaypar["weightsr"]
            weightsh= flaypar["weightsh"]
            
            uzaz= flaypar["uzaz"]
            uzar= flaypar["uzar"]
            uzah= flaypar["uzah"]
            
            biasz= flaypar["biasz"]
            biasr= flaypar["biasr"]
            biash= flaypar["biash"]
            
            ne,te,de= a.shape            
            hbefore= np.zeros((ne,self.big[1]), dtype=np.float32)
            zi = np.empty((ne,te,self.big[1]), dtype=np.float32)
            zdi = np.empty((ne,te,self.big[1]), dtype=np.float32)
            ri = np.empty((ne,te,self.big[1]), dtype=np.float32)
            rdi= np.empty((ne,te,self.big[1]), dtype=np.float32)
            htider =np.empty((ne,te,self.big[1]), dtype=np.float32)
            htiderd = np.empty((ne,te,self.big[1]), dtype=np.float32)
            hi= np.empty((ne,te,self.big[1]), dtype=np.float32)
            
            for k in range(te):
                zi[:,k,:],zdi[:,k,:]= self.activ(a[:,k,:] @ weightsz + hbefore @ uzaz + biasz,"sigmoid")
                ri[:,k,:],rdi[:,k,:]= self.activ(a[:,k,:] @ weightsr + hbefore @ uzar + biasr,"sigmoid")
                htider[:,k,:],htiderd[:,k,:]= self.activ(a[:,k,:] @ weightsh + (ri[:,k,:] * hbefore) @ uzah + biash,"tanh")
                hi[:,k,:]= (1 -zi[:,k,:]) * hbefore + zi[:,k,:] *htider[:,k,:]
                
                hbefore= hi[:,k,:]
                
            co= {"zi": (zi,zdi),"ri": (ri,rdi),"htider": (htider,htiderd),"hi": hi}
            
            return hbefore,co
        
        def backwardgru(self,a,co,flaypar,chan):
            
            uzaz= flaypar["uzaz"]
            uzar= flaypar["uzar"]
            uzah= flaypar["uzah"]
            
            zi,zdi= co["zi"]
            ri,rdi= co["ri"]
            htider,htiderd= co["htider"]
            hi= co["hi"]
            
            ne,te,de= a.shape            
            dweightsz,dweightsr,dweightsh,duzaz,duzar,duzah,dbiasz,dbiasr,dbiash= 0,0,0,0,0,0,0,0,0
        
            for k in reversed(range(te)):                
                hbefore= ifing1(hi,self.big[1],ne,k)    
  
                dzi= chan * (htider[:,k,:] -hbefore) * zdi[:,k,:]
                dhtider= chan * zi[:,k,:] * htiderd[:,k,:]
                dri= (dhtider @ uzah.T) * hbefore * rdi[:,k,:]
                
                dweightsz,duzaz,dbiasz= gruproc(a[:,k,:],dzi,hbefore,dweightsz,duzaz,dbiasz)
                dweightsr,duzar,dbiasr= gruproc(a[:,k,:],dri,hbefore,dweightsr,duzar,dbiasr)
                dweightsh,duzah,dbiash= gruproc(a[:,k,:],dhtider,hbefore,dweightsh,duzah,dbiash)
                
                chan= (chan * (1 -zi[:,k,:])) + (dzi @ uzaz.T) + ((dhtider @ uzah.T) * (ri[:,k,:] + hbefore * (rdi[:,k,:] @ uzar.T)))
            
            return {"weightsz": dweightsz,"uzaz": duzaz,"biasz": dbiasz,"weightsr": dweightsr,"uzar": duzar,"biasr": dbiasr,"weightsh": dweightsh,"uzah": duzah,"biash": dbiash}
              
        def guess(self,a,b=None,accur=True,conf=False):
            
            guessino= self.forwardgo(a)[0]
            
            if not accur:
                return guessino
            
            guessino= guessino.argmax(axis=1)
            b= b.argmax(axis=1)
            
            if not conf:
                return (guessino== b).mean() * 100
            
            cla= np.zeros((len(np.unique(b)),len(np.unique(b))))
            
            for k in range(len(b)):
                cla[b[k]][guessino[k]]= cla[b[k]][guessino[k]] + 1
                
            return cla
        
        def CE(self,d,y):
            return np.sum(np.log(y) * -d) / d.shape[0]
        
    useThisFile= "data.h5"
    filePro= h5py.File(useThisFile,'r')
    
    trainx= np.array(filePro['trX'])
    trainy= np.array(filePro['trY'])
    testx= np.array(filePro['tstX'])
    testy= np.array(filePro['tstY'])
    
    def tryWithRNN():
        """First part -Use RNN"""
        
        print("RNN Layer\n")
        epochRNN= 50
        lrateRNN= 0.01
        batchRNN= 32
        momcRNN= 0.85
        bigRNN= [trainx.shape[2],128,32,16,6]
        
        netrnn,traininglosslrnn,validationlosslrnn,trainingaccuracylrnn,validationaccuracylrnn,testaccuracyrnn= getting(SetNetwork,bigRNN,trainx,trainy,lrateRNN,momcRNN,batchRNN,epochRNN,testx,testy,1)    
        print("\nTest Accuracy: ",testaccuracyrnn,"\n")
        
        trainingconfrnn,testingconfrnn= guessing(netrnn,trainx,trainy,testx,testy)  
        """ Plots"""
        print("First Part - RNN Plots loading... \n\n\n")
        graphn(trainingaccuracylrnn,validationaccuracylrnn,testaccuracyrnn,traininglosslrnn,"RNN")
        plt.title("Training Cross Entropy Loss")
        plt.ylabel("Loss")
        plt.show()
        
        graphn(trainingaccuracylrnn,validationaccuracylrnn,testaccuracyrnn,validationlosslrnn,"RNN")
        plt.title("Validation Cross Entropy Loss")
        plt.ylabel("Loss")
        plt.show()
        
        graphn(trainingaccuracylrnn,validationaccuracylrnn,testaccuracyrnn,trainingaccuracylrnn,"RNN")
        plt.title("Training Accuracy")
        plt.ylabel("Accuracy")
        plt.show()
        
        graphn(trainingaccuracylrnn,validationaccuracylrnn,testaccuracyrnn,validationaccuracylrnn,"RNN")
        plt.title("Validation Accuracy")
        plt.ylabel("Accuracy")
        plt.show()    
        
        graphm(trainingconfrnn,testingconfrnn)
        plt.show()
    
    
    def tryWithLSTM():
        """Second part -Use LSTM"""
        print("LSTM Layer\n")
        
        epochLSTM= 50
        lrateLSTM= 0.01
        batchLSTM= 32
        momcLSTM= 0.85
        bigLSTM= [trainx.shape[2],128,32,16,6]
       
        netlstm,traininglossllstm,validationlossllstm,trainingaccuracyllstm,validationaccuracyllstm,testaccuracylstm= getting(SetNetwork,bigLSTM,trainx,trainy,lrateLSTM,momcLSTM,batchLSTM,epochLSTM,testx,testy,2)
        print("\nTest Accuracy: ",testaccuracylstm,"\n")
    
        trainingconflstm,testingconflstm= guessing(netlstm,trainx,trainy,testx,testy)
        """ Plots"""
        print("Second Part - LSTM Plots loading... \n\n\n")
        graphn(trainingaccuracyllstm,validationaccuracyllstm,testaccuracylstm,traininglossllstm,"LSTM")
        plt.title("Training Cross Entropy Loss")
        plt.ylabel("Loss")
        plt.show()
        
        graphn(trainingaccuracyllstm,validationaccuracyllstm,testaccuracylstm,validationlossllstm,"LSTM")
        plt.title("Validation Cross Entropy Loss")
        plt.ylabel("Loss")
        plt.show()
        
        graphn(trainingaccuracyllstm,validationaccuracyllstm,testaccuracylstm,trainingaccuracyllstm,"LSTM")
        plt.title("Training Accuracy")
        plt.ylabel("Accuracy")
        plt.show()
        
        graphn(trainingaccuracyllstm,validationaccuracyllstm,testaccuracylstm,validationaccuracyllstm,"LSTM")
        plt.title("Validation Accuracy")
        plt.ylabel("Accuracy")
        plt.show()
        
        graphm(trainingconflstm,testingconflstm)
        plt.show()
        
    
    
    def tryWithGRU():
        """Third part - Use GRU"""
    
        print("GRU Layer\n")
        
        epochGRU= 50
        lrateGRU= 0.01
        batchGRU= 32
        momcGRU= 0.85
        bigGRU= [trainx.shape[2],128,32,16,6]
        
        netgru,traininglosslgru,validationlosslgru,trainingaccuracylgru,validationaccuracylgru,testaccuracygru= getting(SetNetwork,bigGRU,trainx,trainy,lrateGRU,momcGRU,batchGRU,epochGRU,testx,testy,3)    
        print("\nTest Accuracy: ",testaccuracygru,"\n")
    
        trainingconfgru,testingconfgru= guessing(netgru,trainx,trainy,testx,testy)
        """ Plots"""
        print("Third Part - GRU Plots loading... \n\n\n")
        graphn(trainingaccuracylgru,validationaccuracylgru,testaccuracygru,traininglosslgru,"GRU")
        plt.title("Training Cross Entropy Loss")
        plt.ylabel("Loss")
        plt.show()
        
        graphn(trainingaccuracylgru,validationaccuracylgru,testaccuracygru,validationlosslgru,"GRU")
        plt.title("Validation Cross Entropy Loss")
        plt.ylabel("Loss")
        plt.show()
        
        graphn(trainingaccuracylgru,validationaccuracylgru,testaccuracygru,trainingaccuracylgru,"GRU")
        plt.title("Training Accuracy")
        plt.ylabel("Accuracy")
        plt.show()
            
        graphn(trainingaccuracylgru,validationaccuracylgru,testaccuracygru,validationaccuracylgru,"GRU")
        plt.title("Validation Accuracy")
        plt.ylabel("Accuracy")
        plt.show()
        
        graphm(trainingconfgru,testingconfgru)
        plt.show()
        
        
    c = 0 
    while c == 0:
        numberr = input("Press 1 for RNN, 2 for LSTM, 3 for GRU ||| Press anything else to exit: ")
        if numberr.isdigit() == True:
            
            if int(numberr) == 1:
                tryWithRNN()
            elif int(numberr) == 2:
                tryWithLSTM()
            elif int(numberr) ==3:
                tryWithGRU()
            else:
                print("\nEXITT !!!")
                c = c - 999999999
                
        else:
            print("\nEXITT !!!")
            c = c - 999999999
            
                
            
            
def getting(n,bi,tr1,tr2,l,m,b,e,te1,te2,no):
    net= n(bi,no)
    trloss,valoss,tracc,valacc= net.workout(tr1,tr2,l,m,b,e).values()
    testacc= net.guess(te1,te2,accur=True)
    return net,trloss,valoss,tracc,valacc,testacc

def guessing(n,tr1,tr2,te1,te2):
    trconf= n.guess(tr1,tr2,accur=True,conf=True)
    teconf= n.guess(te1,te2,accur=True,conf=True)
    return trconf,teconf

def lstmproc(w,b,zic,k):
    w= w + zic.T @ k
    b= b + k.sum(axis=0,keepdims=True)
    return w,b
    
def lstmdu(k,w,big):
    f= k @ w.T[:,:big]
    return f

def gruproc(a,d,h,w,u,b):
        w= w + a.T @ d
        u= u + h.T @ d
        b= b + d.sum(axis=0,keepdims=True)
        return w,u,b

def ifing1(h,b,n,k):
    if k > 0:
        hb= h[:,k-1,:]
    else:
        hb= np.zeros((n,b))
    return hb

def ifing2(h,k):
    if k > 0:
        hb= h[:,k-1,:]
    else:
        hb= 0
    return hb

def graphn(tr,val,te,dat,namee):
    fig= plt.figure(figsize=(20,10))
    fig.suptitle(str(namee)+"\nTraining Accuracy: {:.2f} | Validation Accuracy: {:.2f} | Testing Accuracy: {:.2f}\n ".format(tr[-1],val[-1],te))
    plt.plot(dat)
    plt.xlabel("Epoch")
    
def graphm(trconf,testconf):
    plt.figure(figsize=(20,10),dpi=160)
    plt.subplot(1,2,1)
    sn.heatmap(trconf,annot=True,annot_kws={"size": 8},xticklabels=[1,2,3,4,5,6],yticklabels=[1,2,3,4,5,6],cmap=sn.cm.rocket_r,fmt='g')
    plt.title("Training Confusion Matrix")
    plt.ylabel("Real")
    plt.xlabel("Guess")
    plt.subplot(1,2,2)
    sn.heatmap(testconf,annot=True,annot_kws={"size": 8},xticklabels=[1,2,3,4,5,6],yticklabels=[1,2,3,4,5,6],cmap=sn.cm.rocket_r,fmt='g')
    plt.title("Testing Confusion Matrix")
    plt.ylabel("Real")
    plt.xlabel("Guess")

def calcul(a,b,f,k):
    accu= f(a,b,accur=True)
    loso= k(b,f(a,accur=False))
    return accu,loso

HumActClass()


