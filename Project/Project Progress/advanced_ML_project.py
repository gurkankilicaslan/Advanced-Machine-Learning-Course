# -*- coding: utf-8 -*-
"""
Created on Tue Sept 21 18:37:55 2023

@author: grknk
"""

import h5py
import numpy as np



"""
I wrote the following part for implemantation of RNN.
BUT The majority of following functions could also be used for LSTM and GRU
which are what I am going to implement next.
So, your feedback is appreciated.

Also, should I skip LSTM and GRU, I feel like they will take a lot of time 
if the following RNN implemantation is not true.
should I use other simple methods? 
"""

def RNN():
    class Network():
    
        def __init__(self,big,num):
            
            self.num=num
            self.big=big
            self.laybig= len(big)-1
            self.percbig =None
            self.percpar= None
            self.flaypar =None
            self.percmom= None
            self.flaymom= None
            self.initialize()
            
        def initialize(self):
            big= self.big
            laybig=self.laybig
            
            weights=[]
            bias=[]
            
            for i in range(1,laybig):
                weights.append(np.random.uniform(-(np.sqrt(6)/np.sqrt(big[1]+big[i+1])),(np.sqrt(6)/np.sqrt(big[1]+big[i+1])),size=(big[i],big[i+1])))
                bias.append(np.zeros((1,big[i+1])))
                
            self.percbig =len(weights)
            params={"weights":weights,"bias":bias}
            mom={"weights":[0]*self.percbig,"bias": [0]*len(weights)}
            self.percpar =params
            self.percmom =mom
            
            ne= big[0]
            he= big[1]
            
            
            weightsih= np.random.uniform(-(np.sqrt(6)/np.sqrt(ne+he)),(np.sqrt(6)/np.sqrt(ne+he)),size=(ne,he))
            weightshh= np.random.uniform(-(np.sqrt(6)/np.sqrt(he+he)),(np.sqrt(6)/np.sqrt(he+he)),size=(he,he))
            bias= np.zeros((1,he))
                
            params= {"weightsih": weightsih,"weightshh": weightshh,"bias": bias}
                
           
            mom =dict.fromkeys(params.keys(),0)
            self.flaypar= params
            self.flaymom=mom
            
        def modify(self,lrate,momc,gradflay,gradperc):
            
            flaypar=self.flaypar
            flaymom=self.flaymom
            
            for k in self.flaypar:
                flaymom[k]=lrate*gradflay[k]+momc*flaymom[k]
                flaypar[k]=flaypar[k] - (lrate*gradflay[k]+momc*flaymom[k])
                
            
        def ileriperc(self,a,weights,bias,b):
            return self.activ((a@weights+bias),b)
        
        def geriperc(self,weights,o,der,chan):
            
            dweights=o.T@chan
            dbias=chan.sum(axis=0,keepdims=True)
            chan=der*(chan@weights.T)
            return dweights,dbias,chan 
        
        def workout(self,a,b,lrate,momc,batch,epoch):
            traininglossl,validationlossl,trainingaccuracyl,validationaccuracyl=[],[],[],[]
            
            valbig=int(a.shape[0]/10)
            po=np.random.permutation(a.shape[0])
            vala=a[po][:valbig]
            valb=b[po][:valbig]
            a=a[po][valbig:]
            b=b[po][valbig:]
            it=int(a.shape[0]/batch)
            
            for i in range(epoch):           
                go=0
                stop=batch
                po=np.random.permutation(a.shape[0])
                a=a[po]
                b=b[po]
                
                for j in range(it):
                    gues,o,der,h,hder,co=self.ilerigo(a[go:stop])
                    
                    chan=gues
                    chan[b[go:stop] ==1]=chan[b[go:stop] ==1]-1
                    chan=chan/batch
                    
                    gradflay,gradperc=self.gerigo(a[go:stop],o,der,chan,h,hder,co)
                    
                    self.modify(lrate,momc,gradflay,gradperc)
                    
                    go=stop
                    stop=stop+batch
    
                trainingaccuracy,trainingloss=calcul(a,b,self.guess3,self.CE)                
                validationaccuracy,validationloss=calcul(vala,valb,self.guess3,self.CE)
                    
                print("Epoch: %d === Training Loss: %.3f,Validation Loss: %.3f,Training Accuracy: %.3f,Validation Accuracy: %.3f"% (i+1,trainingloss,validationloss,trainingaccuracy,validationaccuracy))
                traininglossl.append(trainingloss)
                validationlossl.append(validationloss)
                trainingaccuracyl.append(trainingaccuracy)
                validationaccuracyl.append(validationaccuracy)
                
                
                
            return {"traininglossl": traininglossl,"validationlossl": validationlossl,"trainingaccuracyl": trainingaccuracyl,"validationaccuracyl": validationaccuracyl}
        
        def activ(self,a,A):
            if A == "softmax":
                activ=np.exp(a) / np.sum(np.exp(a),axis=1,keepdims=True)
                deriv=None
                return activ,deriv
            
            if A == "tanh":
                activ=np.tanh(a)
                deriv=1 - activ**2
                return activ,deriv
            
            if A == "relu":
                activ=a*(a>0)
                deriv=1*(a>0)
                return activ,deriv
            
            if A == "sigmoid":
                activ=np.exp(a)/(1+np.exp(a))
                deriv=activ*(1-activ)
                return activ,deriv
        
        def ilerirnn(self,a,flaypar):
            
            ne,te,de=a.shape
            
            weightsih=flaypar["weightsih"]
            weightshh=flaypar["weightshh"]
            bias=flaypar["bias"]
            
            hbefore=np.zeros((ne,self.big[1]))
            h,hder=np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1]))
            
            for k in range(te):
                h[:,k,:],hder[:,k,:]=self.activ((a[:,k,:]@weightsih+hbefore@weightshh+bias),"tanh")
                hbefore=h[:,k,:]
                
            return h,hder
        
        def gerirnn(self,a,h,hder,chan,flaypar):
            
            ne,te,de=a.shape            
            weightshh=flaypar["weightshh"]            
            dweightsih,dweightshh,dbias=0,0,0
            
            for k in reversed(range(te)):
                hbefore=ifing1(h,self.big[1],ne,k)
                hbeforeder=ifing2(hder,k)                
                dweightsih=dweightsih+a[:,k,:].T@chan
                dweightshh=dweightshh+hbefore.T@chan
                dbias=dbias+chan.sum(axis=0,keepdims=True)
                chan=hbeforeder*(chan@weightshh)
                
            return {"weightsih": dweightsih,"weightshh": dweightshh,"bias": dbias}
        
        def ilerigo(self,a):
            
            num=self.num
            percpar=self.percpar
            flaypar=self.flaypar
            o,der=[],[]            
            h,hder,co=0,0,0
            
            if num == 1:
                h,hder=self.ilerirnn(a,flaypar)
                o.append(h[:,-1,:])
                der.append(hder[:,-1,:])
            
            if num == 2:
                h,co=self.ilerilstm(a,flaypar)
                o.append(h)
                der.append(1)
                
            if num == 3:
                h,co=self.ilerigru(a,flaypar)
                o.append(h)
                der.append(1)
                
            for i in range(self.percbig-1):
                activ,deriv=self.ileriperc(o[-1],percpar["weights"][i],percpar["bias"][i],"relu")
                o.append(activ)
                der.append(deriv)
            
            gues=self.ileriperc(o[-1],percpar["weights"][-1],percpar["bias"][-1],"softmax")[0]
            
            return gues,o,der,h,hder,co
        
        def gerigo(self,a,o,der,chan,h,hder,co):
            
            num=self.num
            percpar=self.percpar
            flaypar=self.flaypar
            
            gradflay=dict.fromkeys(percpar.keys())
            gradperc={"weights": [0]*self.percbig,"bias": [0]*self.percbig}
            
            for i in reversed(range(self.percbig)):
                gradperc["weights"][i],gradperc["bias"][i],chan=self.geriperc(percpar["weights"][i],o[i],der[i],chan)
                
            if num == 1:
                gradflay=self.gerirnn(a,h,hder,chan,flaypar)
            if num == 2:
                gradflay=self.gerilstm(co,flaypar,chan)
            if num == 3:
                gradflay=self.gerigru(a,co,flaypar,chan)
                
            return gradflay,gradperc
                    
        def ilerilstm(self,a,flaypar):
            
            ne,te,de=a.shape
            
            weightsi,biasi=flaypar["weightsi"],flaypar["biasi"]
            weightsf,biasf=flaypar["weightsf"],flaypar["biasf"]
            weightso,biaso=flaypar["weightso"],flaypar["biaso"]
            weightsc,biasc=flaypar["weightsc"],flaypar["biasc"]
            
            hbefore,cbefore=np.zeros((ne,self.big[1])),np.zeros((ne,self.big[1]))
            zi=np.empty((ne,te,de+self.big[1]))
            hfi=0
            
            hii,hci,hoi,tanhci,ci,tanhcdi,hfdi,hidi,hcdi,hodi=np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1]))
            
            for k in range(te):
                zi[:,k,:]=np.column_stack((hbefore,a[:,k,:]))
                
                hfi,hfdi[:,k,:]=self.activ(zi[:,k,:]@weightsf+biasf,"sigmoid")
                hii[:,k,:],hidi[:,k,:]=self.activ(zi[:,k,:]@weightsi+biasi,"sigmoid")
                hci[:,k,:],hcdi[:,k,:]=self.activ(zi[:,k,:]@weightsc+biasc,"tanh")
                hoi[:,k,:],hodi[:,k,:]=self.activ(zi[:,k,:]@weightso+biaso,"sigmoid")
                
                ci[:,k,:]=hfi*cbefore+hii[:,k,:]*hci[:,k,:]
                tanhci[:,k,:],tanhcdi[:,k,:]=self.activ(ci[:,k,:],"tanh")
                hbefore=hoi[:,k,:]*tanhci[:,k,:]
                cbefore=ci[:,k,:]
                
                co={"zi": zi,"ci": ci,"tanhci": (tanhci,tanhcdi),"hfdi": hfdi,"hii": (hii,hidi),"hci": (hci,hcdi),"hoi": (hoi,hodi)}
                
            return hbefore,co
        
        def gerilstm(self,co,flaypar,chan):
            
            weightsf=flaypar["weightsf"]
            weightsi=flaypar["weightsi"]
            weightsc=flaypar["weightsc"]
            weightso=flaypar["weightso"]
            
            zi=co["zi"]
            ci=co["ci"]
            tanhci,tanhcdi=co["tanhci"]
            hfdi=co["hfdi"]
            hii,hidi=co["hii"]
            hci,hcdi=co["hci"]
            hoi,hodi=co["hoi"]
            te=zi.shape[1]
            
            dweightsf,dweightsi,dweightsc,dweightso,dbiasf,dbiasi,dbiasc,dbiaso=0,0,0,0,0,0,0,0
            
            for k in reversed(range(te)):
                cbefore=ifing2(ci,k)                    
                    
                dci=chan*hoi[:,k,:]*tanhcdi[:,k,:]
                dhfi=dci*cbefore*hfdi[:,k,:]
                dhii=dci*hci[:,k,:]*hidi[:,k,:]
                dhci=dci*hii[:,k,:]*hcdi[:,k,:]
                dhoi=chan*tanhci[:,k,:]*hodi[:,k,:]
                
                dweightsf,dbiasf=lstmproc(dweightsf,dbiasf,zi[:,k,:],dhfi)
                dweightsi,dbiasi=lstmproc(dweightsi,dbiasi,zi[:,k,:],dhii)
                dweightsc,dbiasc=lstmproc(dweightsc,dbiasc,zi[:,k,:],dhci)
                dweightso,dbiaso=lstmproc(dweightso,dbiaso,zi[:,k,:],dhoi)
                
                df=lstmdu(dhfi,weightsf,self.big[1])
                di=lstmdu(dhii,weightsi,self.big[1])
                dc=lstmdu(dhci,weightsc,self.big[1])
                do=lstmdu(dhoi,weightso,self.big[1])
                
                
                chan=(df+di+dc+do)
                
            return {"weightsf": dweightsf,"biasf": dbiasf,"weightsi": dweightsi,"biasi": dbiasi,"weightsc": dweightsc,"biasc": dbiasc,"weightso": dweightso,"biaso": dbiaso}
        
        def ilerigru(self,a,flaypar):
            weightsz=flaypar["weightsz"]
            weightsr=flaypar["weightsr"]
            weightsh=flaypar["weightsh"]
            
            uzaz=flaypar["uzaz"]
            uzar=flaypar["uzar"]
            uzah=flaypar["uzah"]
            
            biasz=flaypar["biasz"]
            biasr=flaypar["biasr"]
            biash=flaypar["biash"]
            
            ne,te,de=a.shape            
            hbefore=np.zeros((ne,self.big[1]))
            zi,zdi,ri,rdi,htider,htiderd,hi=np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1])),np.empty((ne,te,self.big[1]))
            
            for k in range(te):
                zi[:,k,:],zdi[:,k,:]=self.activ(a[:,k,:]@weightsz+hbefore@uzaz+biasz,"sigmoid")
                ri[:,k,:],rdi[:,k,:]=self.activ(a[:,k,:]@weightsr+hbefore@uzar+biasr,"sigmoid")
                htider[:,k,:],htiderd[:,k,:]=self.activ(a[:,k,:]@weightsh+(ri[:,k,:]*hbefore)@uzah+biash,"tanh")
                hi[:,k,:]=(1 - zi[:,k,:])*hbefore+zi[:,k,:]*htider[:,k,:]
                
                hbefore=hi[:,k,:]
                
            co={"zi": (zi,zdi),"ri": (ri,rdi),"htider": (htider,htiderd),"hi": hi}
            
            return hbefore,co
        
        def gerigru(self,a,co,flaypar,chan):
            
            uzaz=flaypar["uzaz"]
            uzar=flaypar["uzar"]
            uzah=flaypar["uzah"]
            
            zi,zdi=co["zi"]
            ri,rdi=co["ri"]
            htider,htiderd=co["htider"]
            hi=co["hi"]
            
            ne,te,de=a.shape            
            dweightsz,dweightsr,dweightsh,duzaz,duzar,duzah,dbiasz,dbiasr,dbiash=0,0,0,0,0,0,0,0,0
        
            for k in reversed(range(te)):                
                hbefore=ifing1(hi,self.big[1],ne,k)    
  
                dzi=chan*(htider[:,k,:] - hbefore)*zdi[:,k,:]
                dhtider=chan*zi[:,k,:]*htiderd[:,k,:]
                dri=(dhtider@uzah.T)*hbefore*rdi[:,k,:]
                
                dweightsz,duzaz,dbiasz=gruproc(a[:,k,:],dzi,hbefore,dweightsz,duzaz,dbiasz)
                dweightsr,duzar,dbiasr=gruproc(a[:,k,:],dri,hbefore,dweightsr,duzar,dbiasr)
                dweightsh,duzah,dbiash=gruproc(a[:,k,:],dhtider,hbefore,dweightsh,duzah,dbiash)
                
                chan=(chan*(1 - zi[:,k,:]))+(dzi@uzaz.T)+((dhtider@uzah.T)*(ri[:,k,:]+hbefore*(rdi[:,k,:]@uzar.T)))
            
            return {"weightsz": dweightsz,"uzaz": duzaz,"biasz": dbiasz,"weightsr": dweightsr,"uzar": duzar,"biasr": dbiasr,"weightsh": dweightsh,"uzah": duzah,"biash": dbiash}
              
        def guess3(self,a,b=None,accur=True,conf=False):
            
            guessino=self.ilerigo(a)[0]
            
            if not accur:
                return guessino
            
            guessino=guessino.argmax(axis=1)
            b=b.argmax(axis=1)
            
            if not conf:
                return (guessino == b).mean()*100
            
            cla=np.zeros((len(np.unique(b)),len(np.unique(b))))
            
            for k in range(len(b)):
                cla[b[k]][guessino[k]]=cla[b[k]][guessino[k]]+1
                
            return cla
        
        def CE(self,d,y):
            return np.sum(np.log(y)*-d) / d.shape[0]
        
    filename="data.h5"
    file=h5py.File(filename,'r')
    
    trainx=np.array(file['trX'])
    trainy=np.array(file['trY'])
    testx=np.array(file['tstX'])
    testy=np.array(file['tstY'])
    
    
    """First part - Use RNN"""

    print("Recurrent Layer\n")
    epoch_rnn=50
    lrate_rnn=0.1
    batch_rnn=32
    momc_rnn=0.85
    bigrnn=[trainx.shape[2],128,32,16,6]
    
    net3rnn,traininglosslrnn,validationlosslrnn,trainingaccuracylrnn,validationaccuracylrnn,testaccuracyrnn=getting(Network,bigrnn,trainx,trainy,lrate_rnn,momc_rnn,batch_rnn,epoch_rnn,testx,testy,1)    
    print("\nTest Accuracy: ",testaccuracyrnn,"\n\n")
    
    trainingconfrnn,testingconfrnn=guessing(net3rnn,trainx,trainy,testx,testy)  
    return trainingconfrnn,testingconfrnn

def getting(n,bi,tr1,tr2,l,m,b,e,te1,te2,no):
    net=n(bi,no)
    trloss,valoss,tracc,valacc=net.workout(tr1,tr2,l,m,b,e).values()
    testacc=net.guess3(te1,te2,accur=True)
    return net,trloss,valoss,tracc,valacc,testacc

def guessing(n,tr1,tr2,te1,te2):
    trconf=n.guess3(tr1,tr2,accur=True,conf=True)
    teconf=n.guess3(te1,te2,accur=True,conf=True)
    return trconf,teconf

def lstmproc(w,b,zic,k):
    w=w+zic.T@k
    b=b+k.sum(axis=0,keepdims=True)
    return w,b
    
def lstmdu(k,w,big):
    f=k@w.T[:,:big]
    return f

def gruproc(a,d,h,w,u,b):
        w=w+a.T@d
        u=u+h.T@d
        b=b+d.sum(axis=0,keepdims=True)
        return w,u,b

def ifing1(h,b,n,k):
    if k>0:
        hb=h[:,k-1,:]
    else:
        hb=np.zeros((n,b))
    return hb

def ifing2(h,k):
    if k>0:
        hb=h[:,k-1,:]
    else:
        hb=0
    return hb


def calcul(a,b,f,k):
    accu=f(a,b,accur=True)
    loso=k(b,f(a,accur=False))
    return accu,loso   
RNN()