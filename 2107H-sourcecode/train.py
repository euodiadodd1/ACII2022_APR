from tqdm import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from scipy import stats
import wandb
import matplotlib.pyplot as plt
import numpy as np

tb = SummaryWriter()

test_loss = 0.0
eval_loss = 0.0
test_mae_loss = 0.0
eval_mae_loss = 0.0
eval_pcc = 0.0
test_pcc = 0.0


def train_model(model, hyp_params, tr_loader, te_loader):
  ## Feel free to remove/setup your own wandb 
    wandb.init(project="my-test-project", entity="euodia")
    wandb.config = {
    "learning_rate": hyp_params.lr,
    "epochs": hyp_params.num_epochs,
    "batch_size": hyp_params.batch_size
    }
    criterion = hyp_params.criterion
    mae = hyp_params.mae
    device= hyp_params.device
    if hyp_params.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.05)
    type = hyp_params.type

    
    for epoch in range(0, hyp_params.num_epochs):
        train_mae_loss = 0
        test_loss = 0
        
        model.train()
        train_predictions = []
        train_labels = []
        test_predictions = []
        test_labels = []
        eval_predictions = []
        eval_labels = []


        for i,(images, labels,p) in enumerate(tr_loader):
          print(i, len(tr_loader))
          labels = labels.to(device).float()
          
          optimizer.zero_grad()

          if type in ["avg_decision", "decision", "attention", "idv_decision"]:
            images = [i.to(device).float() for i in images]
            
            if type == "attention":
                yhat, hidden = model(*images)
            else:
                yhat = model(*images).to(device)
          else:
            images = images.to(device).float()
            yhat = model(images)
            
          train_predictions.append(yhat)
          train_labels.append(labels)
          loss = criterion(yhat, labels)
          loss.backward()
          optimizer.step()

 
        train_predictions = torch.cat(train_predictions)
        train_labels = torch.cat(train_labels)
        train_loss = criterion(train_predictions, train_labels)
        train_mae_loss = mae(train_predictions, train_labels)
        train_pcc = stats.pearsonr(train_predictions.detach().cpu().flatten(), train_labels.detach().cpu().flatten())[0]
      
        print('Epoch: %d | Loss: %.4f | MAE: %.2f | PCC: %.2f'\
            %(epoch, train_loss, train_mae_loss, train_pcc))
        scheduler.step()
        
        model.eval()
        with torch.no_grad(): 
          for i, (images, labels, p) in enumerate(te_loader):

              
              labels = labels.to(device).float()

              if type in ["avg_decision", "decision", "attention", "idv_decision"]:
                images = [i.to(device).float() for i in images]
                
                if type == "attention":
                    outputs, hidden = model(*images)
                else:
                    outputs = model(*images).to(device)
              else:
                images = images.to(device).float()
                outputs = model(images)

              test_predictions.append(outputs)
              test_labels.append(labels)
              loss = criterion(outputs, labels)

          test_predictions = torch.cat(test_predictions)
          test_labels = torch.cat(test_labels)
          test_loss = criterion(test_predictions, test_labels)
          test_mae_loss = mae(test_predictions, test_labels)
          test_pcc = stats.pearsonr(test_predictions.detach().cpu().flatten(), test_labels.detach().cpu().flatten())[0]
          print(test_predictions, test_labels)
          print('Epoch: %d | Test Loss: %.4f | Test MAE Loss: %.2f | Test PCC: %.2f' \
              %(epoch, test_loss, test_mae_loss, test_pcc))
        
          for i, (images, labels, p) in enumerate(tr_loader):
             
              labels = labels.to(device).float()

              if type in ["avg_decision", "decision", "attention", "idv_decision"]:
                images = [i.to(device).float() for i in images]
                if type == "attention":
                    outputs, hidden = model(*images)
                else:
                    outputs = model(*images).to(device)
              else: 
                images = images.to(device).float()
                outputs = model(images)

              eval_predictions.append(outputs)
              eval_labels.append(labels)
              loss = criterion(outputs, labels)
                
          eval_predictions = torch.cat(eval_predictions)
          eval_labels = torch.cat(eval_labels)
          eval_loss = criterion(eval_predictions, eval_labels)
          eval_mae_loss = mae(eval_predictions, eval_labels)
          eval_pcc = stats.pearsonr(eval_predictions.detach().cpu().flatten(), eval_labels.detach().cpu().flatten())[0]
          print('Epoch: %d | Eval Loss: %.4f | Eval MAE Loss: %.2f | Eval PCC: %.2f' \
            %(epoch, eval_loss, eval_mae_loss, eval_pcc))
          

          
        tb.add_scalars("loss", {"Train_loss": eval_loss/len(tr_loader),"Test_loss": test_loss/len(te_loader), "Eval_mae": eval_mae_loss/len(tr_loader), "Test_mae": test_mae_loss/len(te_loader), 
                                "Train_pcc": eval_pcc/len(tr_loader), "Test_pcc": test_pcc/len(te_loader)}, epoch)
        wandb.log({"Train_loss": eval_loss,"Val_loss": test_loss, "Train_mae": eval_mae_loss, "Val_mae": test_mae_loss, 
                                "Train_pcc": eval_pcc, "Val_pcc": test_pcc})
        wandb.watch(model)
    
    print(test_predictions.detach().cpu().flatten(), test_labels.detach().cpu().flatten())
    plot_metrics("train",train_predictions.detach().cpu(), train_labels.detach().cpu())
    plot_metrics("test",test_predictions.detach().cpu(), test_labels.detach().cpu())
    plot_all("all_test",test_predictions.detach().cpu(), test_labels.detach().cpu())
    plot_all("all_train",train_predictions.detach().cpu(), train_labels.detach().cpu())


def plot_metrics(split, preds, labels):
   for i in range(preds.shape[1]):
      plt.scatter(labels[:,i], preds[:,i], c='crimson')
      plt.plot(labels[:,i], labels[:,i])
      plt.xlabel('True Values', fontsize=15)
      plt.ylabel('Predictions', fontsize=15)
      
      plt.savefig(split + "_" + str(i) + ".png")
      plt.show()

def plot_all(split, preds, labels):

    plt.scatter(labels, preds, c='crimson')
    plt.plot(labels, labels)
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    
    plt.savefig(split + ".png")
    plt.show()
