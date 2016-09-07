# Delete all variables defined so far (in notebook)
for name in dir():
    if not callable(globals()[name]) and not name.startswith('_'):
        del globals()[name]

import numpy as np
import os
import mlutils as ml
import neuralnetworks as nn
import qdalda as ql

# from A4mysolution import *

def within(correct, attempt, diff):
    return np.abs((correct-attempt) / correct)  < diff

g = 0

for func in ['trainLDA', 'evaluateLDA',
             'trainQDA', 'evaluateQDA',
             'trainNN', 'evaluateNN']:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined.'.format(func))

data = np.array([
    [13, 3, 1],
    [2, 3, 1],
    [2, 4, 1],
    [13, 4, 1],

    [6, 3, 2],
    [16, 3, 2],
    [5, 2, 2],
    [16, 2, 2],

    [8, 4, 3],
    [18, 4, 3],
    [19, 4, 3],
    [9, 4, 3],
    
    [12, 3, 1],
    [3, 3, 1],
    [12, 4, 1],
    [3, 4, 1],

    [15, 3, 2],
    [5, 3, 2],
    [15, 2, 2],
    [6, 2, 2],

    [18, 3, 3],
    [19, 3, 3],
    [8, 3, 3],
    [9, 3, 3]])

data[:,1] = 3

X = data[:,:2]
T = data[:,2:]



print('''
   Testing   model = trainLDA(X,T)
             accuracy = evaluateLDA(model,X,T)
''')
model = trainLDA(X,T)
acc = evaluateLDA(model,X,T)
if within(50,acc, 10):
    g += 20
    print('20/20 points. Accuracy is within 10 of correct value 50%')
else:
    print(' 0/20 points. Your accuracy of',acc,'is not within 10 of correct value 50%')
    


print('''
   Testing   model = trainQDA(X,T)
             accuracy = evaluateQDA(model,X,T)
''')
model = trainQDA(X,T)
acc = evaluateQDA(model,X,T)
if within(50,acc, 10):
    g += 20
    print('20/20 points. Accuracy is within 10 of correct value 50%')
else:
    print(' 0/20 points. Your accuracy of',acc,'is not within 10 of correct value 50%')
    


print('''
   Testing   model = trainNN(X,T, [[5],100])
             accuracy = evaluateNN(model,X,T)
''')
model = trainNN(X,T, [[5],100])
acc = evaluateNN(model,X,T)
if within(100,acc, 10):
    g += 20
    print('20/20 points. Accuracy is within 10 of correct value 100%')
else:
    print(' 0/20 points. Your accuracy of',acc,'is not within 10 of correct value 100%')
    


print('''\n  Testing
    resultsNN = ml.trainValidateTestKFoldsClassification( trainNN,evaluateNN, X,T, 
                                                          [ [ [0], 5], [ [10], 100] ],
                                                          nFolds=3, shuffle=False,verbose=False)
    bestParms = [row[0] for row in resultsNN]
''')

resultsNN = ml.trainValidateTestKFoldsClassification( trainNN,evaluateNN, X,T, 
                                                      [ [ [0], 5], [ [10], 100] ],
                                                      nFolds=3, shuffle=False,verbose=False)
bestParms = [row[0] for row in resultsNN]
if all([one == [[10],100] for one in bestParms]):
    g += 20
    print('20/20 points. You correctly find the best parameters to be [[10],100] for each fold.')
else:
    print(' 0/20 points. You do not find the correct best parameters to be [[10],100] for each fold.')
    print('              Yours are {}'.format(bestParms))


print('\n{} CODING GRADE is {}/80'.format(os.getcwd().split('/')[-1], g))

print('\n{} WRITING GRADE is ??/20'.format(os.getcwd().split('/')[-1]))

print('\n{} FINAL GRADE is ??/100'.format(os.getcwd().split('/')[-1]))

print('''
Remember, this python script is just an example of how your code will be graded.
Do not be satisfied with an 80% from running this script.  Write and run additional
tests of your own design.''')
