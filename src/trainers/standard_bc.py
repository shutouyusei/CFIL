from .base_trainer import BaseTrainer

class BCTrainer(BaseTrainer):
    def __init__(self,learning_rate=0.001):
        super(BCTrainer,self).__init__(learning_rate=learning_rate)


    def train(self,model,dataset,num_epoches=10,batch_size=32):
        pos_weight = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        print("start training...")
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
        for epoch in range(num_epoches):
            total_loss = 0
            for batch_obs, batch_actions in dataloader:
                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                # Forward 
                predicted_action = model(batch_obs)
                loss = criterion(predicted_action, batch_actions)
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{num_epoches}], Loss: {avg_loss:.4f}")
        print("training is completed.")
        return model
