import sys
import csv
import os
import numpy as np
import datetime
import torch
import torch.nn.functional as F
from utils.io_argparse import get_args
from utils.accuracies import (dev_acc_and_loss, accuracy, approx_train_acc_and_loss)

def normalize(dataset):
  image = np.zeros(dataset.shape)
  for i in range(len(dataset)):
    x = dataset[i]
    std = np.std(x, dtype=np.float64)
    mean = np.mean(x, dtype=np.float64)
    x = (x - mean) / std
    image[i] = x
    x = []
  return image

class BestModel(torch.nn.Module):
    ### TODO Implement your model's structure and input/filter/output dimensions
    def __init__(self, n1_channels, n1_kernel, n2_channels, n2_kernel, pool1,
                 n3_channels, n3_kernel, n4_channels, n4_kernel, pool2, linear_features):
        super().__init__()
        
        self.cov1 = torch.nn.Conv2d(1, n1_channels, n1_kernel)
        self.cov2 = torch.nn.Conv2d(n1_channels, n2_channels, n2_kernel)
        self.p1 = torch.nn.MaxPool2d(pool1)
        self.cov3 = torch.nn.Conv2d(n2_channels, n3_channels, n3_kernel)
        self.cov4 = torch.nn.Conv2d(n3_channels, n4_channels, n4_kernel)
        self.p2 = torch.nn.MaxPool2d(pool2, stride=(2,2))
        self.linear_features = linear_features
        num = int(n4_channels * pow(int((int((28 - n1_kernel - n2_kernel + 2) / pool1) - n3_kernel - n4_kernel + 2) / pool2), 2 ))
        self.linear1 = torch.nn.Linear(num, linear_features)
        self.linear2 = torch.nn.Linear(linear_features, 10)
        
    
    def forward(self, x):
        
        x = torch.reshape(x, (x.shape[0], 1, 28, 28))
        m = torch.nn.ReLU()
        x = self.cov1(x)
        x = m(x)
        x = self.cov2(x)
        x = m(x)
        x = self.p1(x)
        x = self.cov3(x)
        x = m(x)
        x = self.cov4(x)
        x = m(x)
        x = self.p2(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1]))
        return x
    
    

if __name__ == "__main__":
    arguments = get_args(sys.argv)
    MODE = arguments.get('mode')
    DATA_DIR = arguments.get('data_dir')
    
    
    if MODE == "train":
        
        LOG_DIR = arguments.get('log_dir')
        MODEL_SAVE_DIR = arguments.get('model_save_dir')
        LEARNING_RATE = arguments.get('lr')
        BATCH_SIZE = arguments.get('bs')
        EPOCHS = arguments.get('epochs')
        DATE_PREFIX = datetime.datetime.now().strftime('%Y%m%d%H%M')
        if LEARNING_RATE is None: raise TypeError("Learning rate has to be provided for train mode")
        if BATCH_SIZE is None: raise TypeError("batch size has to be provided for train mode")
        if EPOCHS is None: raise TypeError("number of epochs has to be provided for train mode")
        TRAIN_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_images.npy"))
        TRAIN_LABELS = np.load(os.path.join(DATA_DIR, "fruit_labels.npy"))
        DEV_IMAGES = np.load(os.path.join(DATA_DIR, "fruit_dev_images.npy"))
        DEV_LABELS = np.load(os.path.join(DATA_DIR, "fruit_dev_labels.npy"))
        
        ### TODO format your dataset to the appropriate shape/dimensions necessary to be input into your model.

        N_IMAGES, HEIGHT, WIDTH = TRAIN_IMAGES.shape
        N_CLASSES = 6   
        N_DEV_IMGS = len(DEV_IMAGES)
        
        ### TODO Normalize your dataset if desired
        
        flat_train_imgs = normalize(TRAIN_IMAGES)
        flat_dev_imgs = normalize(DEV_IMAGES)
        
        # do not touch the following 4 lines (these write logging model performance to an output file 
        # stored in LOG_DIR with the prefix being the time the model was trained.)
        LOGFILE = open(os.path.join(LOG_DIR, f"bestmodel.log"),'w')
        log_fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc', 'accurate', 'not_accurate']
        logger = csv.DictWriter(LOGFILE, log_fieldnames)
        logger.writeheader()
        
        ### TODO change depending on your model's instantiation
        
        model = BestModel(n1_channels = 8, 
                  n1_kernel = 4, 
                  n2_channels = 8, 
                  n2_kernel = 4, 
                  pool1 = 2,
                  n3_channels = 16, 
                  n3_kernel = 3, 
                  n4_channels = 16, 
                  n4_kernel = 3, 
                  pool2 = 2, 
                  linear_features = 100)
        
        
        ### TODO (OPTIONAL) : you can change the choice of optimizer here if you wish.
        optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
        
    
        for step in range(EPOCHS):
            i = np.random.choice(flat_train_imgs.shape[0], size=BATCH_SIZE, replace=False)
            x = torch.from_numpy(flat_train_imgs[i].astype(np.float32))
            y = torch.from_numpy(TRAIN_LABELS[i].astype(np.int))
            
            
            # Forward pass: Get logits for x
            logits = model(x)
            # Compute loss
            loss = F.cross_entropy(logits, y)
            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # log model performance every 100 epochs
            if step % 100 == 0:
                train_acc, train_loss = approx_train_acc_and_loss(model, flat_train_imgs, TRAIN_LABELS)
                dev_acc, dev_loss, acc, not_acc = dev_acc_and_loss(model, flat_dev_imgs, DEV_LABELS)
                step_metrics = {
                    'step': step, 
                    'train_loss': loss.item(), 
                    'train_acc': train_acc,
                    'dev_loss': dev_loss,
                    'dev_acc': dev_acc,
                    'accurate': acc,
                    'not_accurate': not_acc,


                }

                print(f"On step {step}:\tTrain loss {train_loss}\t|\tDev acc is {dev_acc}")
                
                logger.writerow(step_metrics)
        LOGFILE.close()
        
        ### TODO (OPTIONAL) You can remove the date prefix if you don't want to save every model you train
        ### i.e. "{DATE_PREFIX}_bestmodel.pt" > "bestmodel.pt"
        model_savepath = os.path.join(MODEL_SAVE_DIR,f"bestmodel.pt")
        
        
        print("Training completed, saving model at {model_savepath}")
        torch.save(model, model_savepath)
        
        
    elif MODE == "predict":
        PREDICTIONS_FILE = arguments.get('predictions_file')
        WEIGHTS_FILE = arguments.get('weights')
        if WEIGHTS_FILE is None : raise TypeError("for inference, model weights must be specified")
        if PREDICTIONS_FILE is None : raise TypeError("for inference, a predictions file must be specified for output.")
        TEST_IMAGES = np.load(os.path.join(DATA_DIR, "data.npy"))
        
        model = torch.load(WEIGHTS_FILE)
        
        predictions = []

        TEST_IMAGES = normalize(TEST_IMAGES)
        for test_case in TEST_IMAGES:
            
            ### TODO implement any normalization schemes you need to apply to your test dataset before inference
            
            
            
            x = torch.from_numpy(test_case.astype(np.float32))
            x = x.view(1,-1)
            logits = model(x)
            pred = torch.max(logits, 1)[1]
            predictions.append(pred.item())
        print(f"Storing predictions in {PREDICTIONS_FILE}")
        predictions = np.array(predictions)
        np.savetxt(PREDICTIONS_FILE, predictions, fmt="%d")

    else: raise Exception("Mode not recognized")