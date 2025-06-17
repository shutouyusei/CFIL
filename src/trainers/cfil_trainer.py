from .base_trainer import BaseTrainer

class CFILTrainer(BaseTrainer):
    def __init__(self,learning_rate=0.001):
        super(CFILTrainer,self).__init__(learning_rate)
