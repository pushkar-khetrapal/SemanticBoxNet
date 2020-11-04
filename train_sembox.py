import sys

from datasets import DatasetTrain, DatasetVal # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from semantic_utils.utils import add_weight_decay

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

import time


def train_sem():

    # NOTE! NOTE! change this to not overwrite all log data when you train the model:

    num_epochs = 100
    batch_size = 1
    model_id = '1'
    learning_rate = 0.00001

    train_dataset = DatasetTrain(cityscapes_data_path="data/cityscapes",
                                cityscapes_meta_path="data/cityscapes/meta")
    val_dataset = DatasetVal(cityscapes_data_path="data/cityscapes",
                            cityscapes_meta_path="data/cityscapes/meta")


    num_train_batches = int(len(train_dataset)/batch_size)
    num_val_batches = int(len(val_dataset)/batch_size)
    print ("num_train_batches:", num_train_batches)
    print ("num_val_batches:", num_val_batches)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, shuffle=False,
                                            num_workers=1)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size, shuffle=False,
                                            num_workers=1)

    params = add_weight_decay(model, l2_value=0.0001)
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    with open("data/cityscapes/meta/class_weights.pkl", "rb") as file: # (needed for python3)
        class_weights = np.array(pickle.load(file))
    class_weights = torch.from_numpy(class_weights)
    class_weights = Variable(class_weights.type(torch.FloatTensor)).cuda()

    # loss function
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    epoch_losses_train = []
    epoch_losses_val = []

    model_dir = 'model_weight'


    for epoch in range(len(epoch_losses_val), num_epochs):
        print ("###########################")
        print ("######## NEW EPOCH ########")
        print ("###########################")
        print ("epoch: %d/%d" % (epoch+1, num_epochs))

        ############################################################################
        # train:
        ############################################################################
        with torch.autograd.set_detect_anomaly(True):
        model.train() # (set in training mode, this affects BatchNorm and dropout)
        batch_losses = []
        batch_accuracy_train = []
        current_time = time.time()
        for step, (imgs, label_imgs) in enumerate(train_loader):
            
            imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

            outputs = model(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

            # compute the loss:
            loss = loss_fn(outputs[-1], label_imgs)
            loss_value = loss.data.cpu().numpy() 
            batch_losses.append(loss_value)

            # optimization step:
            optimizer.zero_grad() # (reset gradients)
            #print(loss)
            loss.backward() # (compute gradients)
            optimizer.step() # (perform optimization step)


        print('Time for training 1 epoch : ', time.time() - current_time)
        current_time = time.time()
            

        epoch_loss = np.mean(batch_losses)
        epoch_losses_train.append(epoch_loss)

        
        print ("train loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_train, "k^")
        plt.plot(epoch_losses_train, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("train loss per epoch")
        plt.savefig("%s/epoch_losses_train.png" % model_dir)
        plt.close(1)


        print ("####")

        ############################################################################
        # val:
        ############################################################################
        model.eval() # (set in evaluation mode, this affects BatchNorm and dropout)
        batch_losses = []
        batch_accuracy_test = []
        for step, (imgs, label_imgs, img_ids) in enumerate(val_loader):
            with torch.no_grad(): # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
                imgs = Variable(imgs).cuda() # (shape: (batch_size, 3, img_h, img_w))
                label_imgs = Variable(label_imgs.type(torch.LongTensor)).cuda() # (shape: (batch_size, img_h, img_w))

                outputs = model(imgs) # (shape: (batch_size, num_classes, img_h, img_w))

                intersection = np.logical_and(label_imgs.data.cpu().numpy(), outputs[-1].data.cpu().numpy())
                union = np.logical_or(label.data.cpu().numpy(), outputs[-1].data.cpu().numpy())
                iou_score = np.sum(intersection) / np.sum(union)
                batch_accuracy_test.append(iou_score)


                # compute the loss:
                loss = loss_fn(outputs[-1], label_imgs)
                loss_value = loss.data.cpu().numpy()
                batch_losses.append(loss_value)

            

        epoch_loss = np.mean(batch_losses)
        epoch_losses_val.append(epoch_loss)
        
        print('Time for testing 1 epoch : ', time.time() - current_time)
        
        print ("val loss: %g" % epoch_loss)
        plt.figure(1)
        plt.plot(epoch_losses_val, "k^")
        plt.plot(epoch_losses_val, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("val loss per epoch")
        plt.savefig("%s/epoch_losses_val.png" % model_dir)
        plt.close(1)



        # save the model weights to disk:
        checkpoint_path = model_dir + "/model_" + model_id +"_epoch_" + str(epoch+1) + ".pth"
        torch.save(model.state_dict(), checkpoint_path)

        np.save('model_weight/training_loss.npy', epoch_losses_train)
        np.save('model_weight/validation_loss.npy', epoch_losses_val)

if __name__ == "__main__":
    train_sem()