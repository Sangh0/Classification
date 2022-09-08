import time

import torch
import torch.nn as nn
import torch.optim as optim


class TrainModel(object):
    
    def __init__(
        self,
        model,
        lr=0.01,
        epochs=90,
        weight_decay=0.0005,
        lr_scheduling=True,
        train_log_step=10,
        valid_log_step=5,
    ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        
        self.loss_func = nn.CrossEntropyLoss()
        
        self.epochs = epochs
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            momentum=0.9,
            lr=lr,
            weight_decay=weight_decay,
        )
        
        self.lr_scheduling = lr_scheduling
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=True,
        )
        
        self.train_log_step = train_log_step
        self.valid_log_step = valid_log_step
        
    def fit(self, train_data, validation_data):
        print('Start Model Training...!')
        start_training = time.time()
        pbar = tqdm(range(self.epochs), total=int(self.epochs))
        for epoch in pbar:
            init_time = time.time()
            
            ######### Train Phase #########
            train_loss, train_acc = self.train_on_batch(
                train_data, self.train_log_step,
            )
            
            ######### Validate Phase #########
            valid_loss, valid_acc = self.validate_on_batch(
                validation_data, self.valid_log_step,
            )
            
            end_time = time.time()
            
            print(f'\n{"="*30} Epoch {epoch+1}/{self.epochs} {"="*30}'
                  f'\ntime: {end_time-init_time:.2f}s'
                  f'   lr = {self.optimizer.param_groups[0]["lr"]}')
            print(f'\ntrain average loss: {train_loss:.3f}'
                  f'  accuracy: {train_acc:.3f}')
            print(f'\nvalid average loss: {valid_loss:.3f}'
                  f'  accuracy: {valid_loss:.3f}')
            print(f'\n{"="*80}')
            
            if self.lr_scheduling:
                self.lr_scheduler.step(valid_loss)
                
        end_training = time.time()
        print(f'\nTotal time for training is {end_training-start_training:.2}s')
        
        return {
            'model': self.model,
        }
        
        
    @torch.no_grad()
    def validate_on_batch(self, validation_data, log_step):
        self.model.eval()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(validation_data):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index==labels).sum()/len(outputs)
            
            if batch == 0:
                print(f'\n{" "*10} Validate Step {" "*10}')
                
            if (batch+1) % log_step == 0:
                print(f'\n[Batch {batch+1}/{len(validation_data)}]'
                      f'  valid loss: {loss:.3f}  accuracy: {acc:.3f}')
                
            batch_loss += loss.item()
            batch_acc += acc.item()
            
            del images; del labels; del outputs
            torch.cuda.empty_cache()
            
        return batch_loss/(batch+1), batch_acc/(batch+1)
        
        
    def train_on_batch(self, train_data, log_step):
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(train_data):
            self.model.train()
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.loss_func(outputs, labels)
            output_index = torch.argmax(outputs, dim=1)
            acc = (output_index==labels).sum()/len(outputs)
            
            loss.backward()
            self.optimizer.step()
            
            if batch == 0:
                print(f'\n{" "*10} Train Step {" "*10}')
                
            if (batch+1) % log_step == 0:
                print(f'\n[Batch {batch+1}/{len(train_data)}]'
                      f'  train loss: {loss:.3f}  accuracy: {acc:.3f}')
                
            batch_loss += loss.item()
            batch_acc += acc.item()
            
            del images; del labels; del outputs
            torch.cuda.empty_cache()
        
        return batch_loss/(batch+1), batch_acc/(batch+1)