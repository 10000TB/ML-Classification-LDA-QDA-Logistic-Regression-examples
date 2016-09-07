# Delete all variables defined so far (in notebook)
for name in dir():
    if not callable(globals()[name]) and not name.startswith('_'):
        del globals()[name]

import numpy as np
import os

# import A2mysolution as mine
# import imp
# imp.reload(mine)
# makeMPGData = mine.makeMPGData
# findBestPower = mine.findBestPower

def within(correct, attempt, diff):
    return np.abs((correct-attempt) / correct)  < diff

g = 0

if 'findBestPower' not in dir() or not callable(globals()['findBestPower']):
    print('CRITICAL ERROR: Function named \'findBestPower\' is not defined.')
if 'makeMPGData' not in dir() or not callable(globals()['makeMPGData']):
    print('CRITICAL ERROR: Function named \'makeMPGData\' is not defined.')

X,T,Xnames,Tname = makeMPGData()


def testPower(nFolds,maxPowers,correctValue,withinValue):
    power,avgs = findBestPower(X,T,nFolds,maxPowers)
    if not within(correctValue,power,withinValue):
        print(' 0/20 points. \'findBestPower(X,T,nFolds={:d},maxPowers={})\' returns {:d} but should return {:d}'.format(nFolds,maxPowers,power,correctValue))
        return 0
    else:
        print('20/20 points. \'findBestPower(X,T,nFolds={:d},maxPowers={})\' correctly returns {:d}'.format(nFolds,maxPowers,correctValue))
        return 20

g += testPower(20,[1],1,0.1)
g += testPower(20,range(1,4),3,0.1)
g += testPower(20,range(1,10),3,0.1)
    
_,avgs = findBestPower(X,T,10,range(1,10))
correctAvgs = np.array([[1,   3.2798,   3.4234],
                        [2,   2.6881,   2.8317],
                        [3,   2.5108,   2.6863],
                        [4,   2.4724,   2.7695],
                        [5,   2.4578,   2.8630],
                        [6,   2.4620,   3.3136],
                        [7,   2.5565,   10.8798],
                        [8,   2.4316,   3.6759],
                        [9,   2.4949,   7.6110]])


if np.abs(avgs[:4,:]-correctAvgs[:4,:]).sum() > 5:
    print(' 0/20 points. \'In power,avgs = findBestPower(X,T,nFolds=10,maxPowers=range(1,10))\' your \'avgs\' array:')
    for row in avgs:
        print('   {:.0f}   {:.4f}   {:.4f}'.format(*row))
    print(' does not match the following correct array:')
    for row in correctAvgs:
        print('   {:.0f}   {:.4f}   {:.4f}'.format(*row))
else:
    g += 20
    print('20/20 points. \'In power,avgs = findBestPower(X,T,nFolds=10,maxPowers=range(1,10))\', your \'avgs\' array is correct.')

print('\n{} Grade is {}/100'.format(os.getcwd().split('/')[-1], g))
print('Up to 20 more points will be given based on the qualty of your descriptions of the method and the results.')
