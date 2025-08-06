
class BaseModel:
    def __init__(self):
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def predict_one(self):
        raise NotImplementedError()
    
    def predict(self):
        raise NotImplementedError()
