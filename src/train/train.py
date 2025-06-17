from . data_load import Data 
import trainers
import networks
import torch

class TrainBase:
    def train(self,save_path):
        raise NotImplementedError

class TrainBC(TrainBase):
    def __init__(self):
        self.model = networks.MarioNetwork() 
        self.trainer = trainers.BCTrainer(learning_rate=0.001)
        self.dataset = Data().load(10,[1])

    def train(self,save_path):
        self.model = trainer.train(self.model,self.dataset)
        torch.save(self.model.state_dict(), save_path)

class TrainGAIL(TrainBase):
    def __init__(self):
        self.trainer = trainers.GAILTrainer(learning_rate=0.001,gamma=0.99,gail_lambda=0.1)
        self.obs_data, self.action_data = Data().load_data(10,[1])
        self.policy_network = networks.PolicyNetwork(3,6,64)
        self.discriminator = networks.Discriminator(172032,6,64)

    def train(self,save_path):
        self.trainer.train(self.policy_network, self.discriminator, self.obs_data,self.action_data, num_epoches=10,batch_size=32,policy_iterations=5,discriminator_iterations=1)
        torch.save(self.policy_network.state_dict(), save_path)
