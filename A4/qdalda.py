import numpy as np
import sys  # for sys.float_info.epsilon

######################################################################
### class QDA
######################################################################

class QDA(object):
    
    def __init__(self):
        # Define all instance variables here. Not necessary
        self.Xmeans = None
        self.Xstds = None
        self.mu = None
        self.sigma = None
        self.sigmaInv = None
        self.prior = None
        self.discriminantConstant = None

    def train(self,X,T):
        self.classes = np.unique(T)
        self.Xmeans = np.mean(X,0)
        self.Xstds = np.std(X,0)
        self.Xconstant = self.Xstds == 0
        self.XstdsFixed = self.Xstds.copy()
        self.XstdsFixed[self.Xconstant] = 1
        Xs = (X - self.Xmeans) / self.XstdsFixed
        self.mu = []
        self.sigma = []
        self.sigmaInv = []
        self.prior = []
        nSamples = X.shape[0]
        for k in self.classes:
            rowsThisClass = (T == k).reshape((-1))
            self.mu.append( np.mean(Xs[rowsThisClass,:],0).reshape((-1,1)) )
            if sum(rowsThisClass) == 1:
                self.sigma.append(np.eye(Xs.shape[1]))
            else:
                self.sigma.append( np.cov(Xs[rowsThisClass,:],rowvar=0) )
            if self.sigma[-1].size == 1:
                self.sigma[-1] = self.sigma[-1].reshape((1,1))
            self.sigmaInv.append( np.linalg.pinv(self.sigma[-1]) )    # pinv in case Sigma is singular
            self.prior.append( np.sum(rowsThisClass) / float(nSamples) )
        self._finishTrain()

    def _finishTrain(self):
        self.discriminantConstant = []
        for ki in range(len(self.classes)):
            determinant = np.linalg.det(self.sigma[ki])
            if determinant == 0:
                # raise np.linalg.LinAlgError('trainQDA(): Singular covariance matrix')
                determinant = sys.float_info.epsilon
            self.discriminantConstant.append( np.log(self.prior[ki]) - 0.5*np.log(determinant) )

    def use(self,X):
        nSamples = X.shape[0]
        Xs = (X - self.Xmeans) / self.XstdsFixed
        discriminants = self._discriminantFunction(Xs)
        predictedClass = self.classes[np.argmax( discriminants, axis=1 )]
        predictedClass = predictedClass.reshape((-1,1))
        D = X.shape[1]
        probabilities = np.exp( discriminants - 0.5*D*np.log(2*np.pi) - np.log(np.array(self.prior)) )
        return predictedClass,probabilities,discriminants

    def _discriminantFunction(self,Xs):
        nSamples = Xs.shape[0]
        discriminants = np.zeros((nSamples, len(self.classes)))
        for ki in range(len(self.classes)):
            Xc = Xs - self.mu[ki].reshape((-1))
            discriminants[:,ki:ki+1] = self.discriminantConstant[ki] - 0.5 * \
                                       np.sum(np.dot(Xc, self.sigmaInv[ki]) * Xc, axis=1).reshape((-1,1))
        return discriminants
        
    def __repr__(self):
        if self.mu is None:
            return 'QDA not trained.'
        else:
            return 'QDA trained for classes {}'.format(self.classes)

######################################################################
### class LDA
######################################################################

class LDA(QDA):

    def _finishTrain(self):
        self.sigmaMean = np.sum(np.stack(self.sigma) * np.array(self.prior)[:,np.newaxis,np.newaxis], axis=0)
        self.sigmaMeanInv = np.linalg.pinv(self.sigmaMean)
        # print(self.sigma)
        # print(self.sigmaMean)
        self.discriminantConstant = []
        self.discriminantCoefficient = []
        for ki in range(len(self.classes)):
            sigmaMu = np.dot(self.sigmaMeanInv, self.mu[ki])
            self.discriminantConstant.append( -0.5 * np.dot(self.mu[ki].T, sigmaMu) )
            self.discriminantCoefficient.append( sigmaMu )
    
    def _discriminantFunction(self,Xs):
        nSamples = Xs.shape[0]
        discriminants = np.zeros((nSamples, len(self.classes)))
        for ki in range(len(self.classes)):
            discriminants[:,ki:ki+1] = self.discriminantConstant[ki] + \
                                       np.dot(Xs, self.discriminantCoefficient[ki])
        return discriminants

    def __repr__(self):
        if self.mu is None:
            return 'LDA not trained.'
        else:
            return 'LDA trained for classes {}'.format(self.classes)


######################################################################
### Example use
######################################################################

if __name__ == '__main__':

    D = 1  # number of components in each sample
    N = 10  # number of samples in each class
    X = np.vstack((np.random.normal(0.0,1.0,(N,D)),
                   np.random.normal(4.0,1.5,(N,D))))
    T = np.vstack((np.array([1]*N).reshape((N,1)),
                   np.array([2]*N).reshape((N,1))))

    qda = QDA()
    qda.train(X,T)
    c,prob,d = qda.use(X)
    print('QDA', np.sum(c==T)/X.shape[0] * 100, '% correct')
    # print(np.hstack((T,c)))
    # print(prob)
    # print(d)

    lda = LDA()
    lda.train(X,T)
    c,prob,d = lda.use(X)
    print('LDA', np.sum(c==T)/X.shape[0] * 100, '% correct')
    # print(np.hstack((T,c)))
    # print(prob)
    # print(d)
