import pickle
import numpy as np
import time

import extratrees

X = np.array([[1, 2], [3, 3], [2, 4]], dtype=np.float32, order='F')
y = np.array([1, 2, 3], dtype=np.float64)


forest = extratrees.train(X, y, number_of_trees=100, regression=True)

start = time.time()
pickle_data = pickle.dumps(forest)
forest2 = pickle.loads(pickle_data)
stop = time.time()

print "pickling + unpickling took %.3f secs" % (stop - start)


def test_pickle_and_repickle():
    pickle_data2 = pickle.dumps(forest2)
    assert pickle_data == pickle_data2, (pickle_data, pickle_data2)


def test_pickle_prediction():
    vectors = np.array([[1.1, 2]], dtype=np.float32)
    pred1 = forest.predict(vectors)
    pred2 = forest2.predict(vectors)
    assert pred1 == pred2
