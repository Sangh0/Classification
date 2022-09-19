import time
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from util.callback import CheckPoint

class TrainModel(object):
    
    def __init__(
        self,
        model,
        lr=0.01,
        epochs=90,
        weight_decay=0.0005,
        lr_scheduling=True,
        check_point=True,
        train_log_step=10,
        valid_log_step=5,
    ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ######### Parallel GPU training #########
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)
            self.model = model.to(self.device)

        ######### Single GPU training #########
        else:
            print('Single GPU training!')
            self.model = model.to(self.device)

        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        
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
            min_lr=lr*1e-2,
        )

        self.check_point = check_point
        self.cp = CheckPoint(verbose=True)
        
        self.train_log_step = train_log_step
        self.valid_log_step = valid_log_step
        
    def fit(self, train_data, validation_data):
        print('Start Model Training...!')
        loss_list, acc_list = [], []
        val_loss_list, val_acc_list = [], []
        start_training = time.time()
        pbar = tqdm(range(self.epochs), total=int(self.epochs))
        for epoch in pbar:
            init_time = time.time()
            
            ######### Train Phase #########
            train_loss, train_acc = self.train_on_batch(
                train_data, self.train_log_step,
            )

            loss_list.append(train_loss)
            acc_list.append(train_acc)
            
            ######### Validate Phase #########
            valid_loss, valid_acc = self.validate_on_batch(
                validation_data, self.valid_log_step,
            )
            
            val_loss_list.append(valid_loss)
            val_acc_list.append(valid_acc)

            end_time = time.time()
            
            print(f'\n{"="*30} Epoch {epoch+1}/{self.epochs} {"="*30}'
                  f'\ntime: {end_time-init_time:.2f}s'
                  f'   lr = {self.optimizer.param_groups[0]["lr"]}')
            print(f'\ntrain average loss: {train_loss:.3f}'
                  f'  accuracy: {train_acc:.3f}')
            print(f'\nvalid average loss: {valid_loss:.3f}'
                  f'  accuracy: {valid_acc:.3f}')
            print(f'\n{"="*80}')
            
            if self.lr_scheduling:
                self.lr_scheduler.step(valid_loss)

            if self.check_point:
                path = f'./weights/check_point_{epoch+1}.pt'
                self.cp(valid_loss, self.model, path)

        end_training = time.time()
        print(f'\nTotal time for training is {end_training-start_training:.2}s')
        
        return {
            'model': self.model,
            'loss': loss_list,
            'acc': acc_list,
            'val_loss': val_loss_list,
            'val_acc': val_acc_list,
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
        self.model.train()
        batch_loss, batch_acc = 0, 0
        for batch, (images, labels) in enumerate(train_data):
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