# Delete all variables defined so far (in notebook)
for name in dir():
    if not callable(globals()[name]) and not name.startswith('_'):
        del globals()[name]

import numpy as np
#import sklearn.datasets as skdata
import types
import os


X = np.array([[4,2],
              [5,1],
              [8,4],
              [2,1],
              [4,6],
              [7,3]])
T = X[:,0:1] * 5 + X[:,1:2] * -1 + 2


model = train(X,T)
predict = use(X,model)
error = rmse(predict,T)

def within(a,b, diff):
    return np.abs(a-b) < diff

g = 0

if 'means' not in model.keys():
    print(' 0/20 points. \'train\' does not return dict with a \'means\' key.')
else:
    means = model['means']
    if len(means) != 2:
        print(' 0/20 points. \'means\' should be length 2. It is length',len(means))
    else:
        if not within(means[0],5,0.1) or not within(means[1],2.83,0.1):
            print(' 0/20 points. \'means\' values are not correct.')
        else:
            g += 20
            print('20/20 points. \'means\' values are correct.')
    
if 'stds' not in model.keys():
    print(' 0/20 points. \'train\' does not return dict with a \'stds\' key.')
else:
    stds = model['stds']
    if len(stds) != 2:
        print(' 0/20 points. \'stds\' should be length 2. It is length',len(stds))
    else:
        if not within(stds[0],2.0,0.1) or not within(stds[1],1.77,0.1):
            print(' 0/20 points. \'stds\' values are not correct.')
        else:
            g += 20
            print('20/20 points. \'stds\' values are correct.')
    
if 'w' not in model.keys():
    print(' 0/20 points. \'train\' does not return dict with a \'w\' key.')
else:
    w = model['w'] 
    if len(w) != 3:
        print(' 0/20 points. \'w\' should be length 3. It is length',len(w))
    else:
        if not within(w[0],24.17,0.2) or not within(w[1],10,0.2) or not within(w[2],-1.77,0.2):
            print(' 0/20 points. \'w\' values are not correct.')
        else:
            g += 20
            print('20/20 points. \'w\' values are correct.')
        
if not within(predict[0],20,2) or not within(predict[2],38,2) or not within(predict[4],16,2):
    print(' 0/20 points. Values returned by \'use\' not correct.')
else:
    g += 20
    print('20/20 points. Values returned by \'use\' are correct.')
    
if within(error,7.2e-15,1e-4):
    g += 20
    print('20/20 points. rmse() is correct.')
else:
    print(' 0/10 points. rmse() not correct. It returned',error,'but should be',7.2e-15)

print('{} Grade is {}/100'.format(os.getcwd().split('/')[-1], g))

# print('means',model['means'])
# print('stds',model['stds'])
# print(np.hstack((predict[:4,:],T[:4,:])))
# print(error)
