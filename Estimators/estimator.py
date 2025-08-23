
class Estimator:
    '''
    Base class for all estimators.
    '''
    def __init__(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()
