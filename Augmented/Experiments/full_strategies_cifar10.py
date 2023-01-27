# model: vgg16, pretrained with imgnet
# dataset: cifar10 
# augmanetations: 2
# results: tensorboard, csv


# strategy: uncertainties over all elements
# labeling: all elements of one image  
import os
import pickle
import argparse
import sys
from pprint import pprint
import random
import csv
from copy import deepcopy
from time import time
import datetime
import numpy as np
import pandas as pd

from tensorboardX import SummaryWriter
import torch
import torch.backends
from torch import optim
from torch.hub import load_state_dict_from_url
from torch.nn import CrossEntropyLoss
from torchvision import datasets
from torchvision.models import vgg16
from torchvision.transforms import transforms
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import patch_module
from baal import ModelWrapper
from baal.utils.metrics import Accuracy
from baal.active.heuristics import BALD
from baal.active.dataset import ActiveLearningDataset

import aug_lib

from baal_extended.ExtendedActiveLearningDataset import ExtendedActiveLearningDataset


"""
Minimal example to use BaaL.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--initial_pool", default=1000, type=int)
    parser.add_argument("--query_size", default=100, type=int)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--heuristic", default="bald", type=str)
    parser.add_argument("--iterations", default=20, type=int)
    parser.add_argument("--shuffle_prop", default=0.05, type=float)
    parser.add_argument("--learning_epoch", default=20, type=int)
    parser.add_argument("--strategy", default="all", type=str)
    parser.add_argument("--tag", default="notag", type=str)
    return parser.parse_args()


def get_datasets(initial_pool):
    transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    aug_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            aug_lib.TrivialAugment(),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    test_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(3 * [0.5], 3 * [0.5]),
        ]
    )
    # Note: We use the test set here as an example. You should make your own validation set.
    train_ds = datasets.CIFAR10(
        ".", train=True, transform=transform, target_transform=None, download=True
    )

    aug_train_ds = datasets.CIFAR10(
        ".", train=True, transform=aug_transform, target_transform=None, download=True
    )
    test_set = datasets.CIFAR10(
        ".", train=False, transform=test_transform, target_transform=None, download=True
    )
    eald_set = ExtendedActiveLearningDataset(train_ds)
    # active_set = ActiveLearningDataset(
    #    train_ds, pool_specifics={"transform": test_transform}
    # )
    eald_set.augment_n_times(2, augmented_dataset=aug_train_ds)
    # We start labeling randomly.
    eald_set.label_randomly(initial_pool)
    return eald_set, test_set

# save uncertainties in pickle file
def generate_pickle_file(tag, dt_string, active_set, epoch, oracle_indices, uncertainty):    
    pickle_filename = dt_string + (
        f"_uncertainty_epoch={epoch}" f"_labelled={len(active_set)}.pkl"
    )
    dir_path = os.path.join(os.getcwd(), "uncertainties")
    isExist = os.path.exists("uncertainties")
    if not isExist:
        os.makedirs(dir_path)
    pickle_file_path = os.path.join(dir_path, pickle_filename)
    print("Saving file " + pickle_file_path)
    pickle.dump(
        {
            "oracle_indices": oracle_indices,
            "uncertainty": uncertainty,
            "labelled_map": active_set.labelled_map,
        },
        open(pickle_file_path, "wb")
    )
    return dir_path, pickle_file_path

def main():
    args = parse_args()
    hyperparams = vars(args)
    tag = hyperparams["tag"]
    strategy = hyperparams["strategy"]
    use_cuda = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    random.seed(1337)
    torch.manual_seed(1337)
    if not use_cuda:
        print("warning, the experiments would take ages to run on cpu")

    now = datetime.datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%Hx%M")
    with open(f"results/csv/metrics_cifarnet_{tag}_{dt_string}_.csv", "w+", newline="") as out_file:
        csvwriter = csv.writer(out_file)
        csvwriter.writerow(
            (
                "epoch",
                "test_acc",
                "train_acc",
                "test_loss",
                "train_loss",
                "Next training size",
                "amount original images labelled",
                "amount augmented images labelled"
            )
        )

        active_set, test_set = get_datasets(hyperparams["initial_pool"])

        heuristic = get_heuristic(hyperparams["heuristic"], hyperparams["shuffle_prop"])
        criterion = CrossEntropyLoss()
        model = vgg16(weights="VGG16_Weights.DEFAULT")
        #anpassen Model an zehn Klassen
        model.classifier[6] = nn.Linear(4096, 10)

        # change dropout layer to MCDropout
        model = patch_module(model)

        if use_cuda:
            model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=hyperparams["lr"], momentum=0.9)

        # Wraps the model into a usable API.
        model = ModelWrapper(model, criterion, replicate_in_memory=False)
        model.add_metric(name='accuracy',initializer=lambda : Accuracy())

        logs = {}
        logs["epoch"] = 0

        # for prediction we use a smaller batchsize
        # since it is slower

        # We will reset the weights at each active learning step.
        init_weights = deepcopy(model.state_dict())

        layout = {
            "Loss/Accuracy": {
                "Loss": ["Multiline", ["loss/train", "loss/test"]],
                "Accuracy": ["Multiline", ["accuracy/train", "accuracy/test"]],
                },
        }

        chkp_path = f"results/checkpoints/model_{tag}_{dt_string}.pt"

        tensorboardwriter = SummaryWriter(f"results/tensorboard/{tag}_{dt_string}")
        tensorboardwriter.add_custom_scalars(layout)

        n_augmented_old = 0
        n_original_old = 0

        for epoch in tqdm(range(args.epoch)):
            # if we are in the last round we want to train for longer epochs to get a more comparable result
            if epoch == (args.epoch - 1):
                hyperparams["learning_epoch"] = 75
            
            # Load the initial weights.
            model.load_state_dict(init_weights)
            model.train_on_dataset(
                active_set,
                optimizer,
                hyperparams["batch_size"],
                hyperparams["learning_epoch"],
                use_cuda,
            )

            # Validation!
            model.test_on_dataset(test_set, hyperparams["batch_size"], use_cuda)
            metrics = model.metrics
            
            # get origin amount of labelled augmented/unaugmented images
            if(epoch == 0):
                csvwriter.writerow(
                    (
                        -1,
                        0,
                        0,
                        0,
                        0,
                        active_set.n_labelled,
                        active_set.n_unaugmented_images_labelled,
                        active_set.n_augmented_images_labelled
                    )
                )
            # switch for strategy
            if (strategy == "all"):
                
                active_loop = ActiveLearningLoop(
                    active_set,
                    model.predict_on_dataset,
                    heuristic,
                    hyperparams.get("query_size", 1),
                    # save uncertainties into one file per epoch
                    #uncertainty_folder="results/uncertainties",
                    batch_size=10,
                    iterations=hyperparams["iterations"],
                    use_cuda=use_cuda,
                )
                should_continue = active_loop.step()
                if not should_continue:
                    break

            elif (strategy == "mean"):
                
                orgset_len = int(len(active_set._dataset)/(3)) # length of original dataset (dataset divided through amount augmentatios + 1)
                print("orgset_len: " + str(orgset_len))
                pool = active_set.pool
                if len(pool) > 0:
                    indices = np.arange(len(pool)) # array von 0 bis len(pool)-1 (nach initial label: 146999)
                    probs = model.predict_on_dataset(pool,batch_size=10,iterations=hyperparams["iterations"],use_cuda=use_cuda)
                    #if probs is not None and (isinstance(probs, types.GeneratorType) or len(probs) > 0):
                    # -> "isinstance(...) needed when using predict_..._Generator"
                    if probs is not None and len(probs) > 0:
                        #to_label, uncertainties = heuristic.get_ranks(probs) 
                        uncertainties = heuristic.get_uncertainties(probs)
                            # to_label -> indices sortiert von größter zu niedrigster uncertainty
                            # uncertainty -> alle uncertainties des pools in Reihenfolge wie pool vorliegt
                            #to_label = indices[np.array(to_label)] # was hier passiert keine Ahnung aber to_label bleibt gleich also unnütze Zeile?
                        # 1. get all original images from to_label(whole pool sorted after uncertainties highest to lowest)
                        # 2. get for each original image the ids of augmented children and all three uncertainties
                        # 3. calculate the means
                        # 4. sort those means after highest to lowest uncertainties 
                        # 5. label images (all of one kind) after query value with pool indices
                        trios = []
                        oracle_idx = active_set._pool_to_oracle_index(indices)
                        for idx in oracle_idx:   
                            # checks if img already in trios
                            if idx not in trios:
                                if idx >= orgset_len and idx < len(active_set._dataset)-orgset_len:
                                    # img is first augmentation
                                    org = idx - orgset_len
                                    aug1 = idx
                                    aug2 = idx + orgset_len
                                elif idx >= len(active_set._dataset)-orgset_len:
                                    # img is second augmentation
                                    org = idx - 2*orgset_len
                                    aug1 = idx - orgset_len
                                    aug2 = idx
                                else:
                                    # img is original 
                                    org = idx
                                    aug1 = idx + orgset_len
                                    aug2 = idx + 2*orgset_len

                                trios.append(int(org))
                                trios.append(int(aug1))
                                trios.append(int(aug2))
                        trios_uncertainties = []
                        pool_trios = active_set._oracle_to_pool_index(trios)
                        for i in range(len(pool_trios)):
                            # get the uncertainty of all images
                            trios_uncertainties.append(uncertainties[pool_trios[i]])
                        trios_mean = []
                        k = trios_uncertainties[0]
                        assert len(trios_uncertainties) % 3 == 0, "trios_mean should be a multiple of augmentations + 1"
                        for i in range(1,len(trios_uncertainties)):
                            if (i+1) % 3 != 0:
                                k += trios_uncertainties[i]
                            else:
                                k += trios_uncertainties[i]
                                mean = k/3
                                trios_mean.append(mean)
                                k = 0
                        trios_idx_mean_sorted = np.argsort(trios_mean)[::-1]
                        to_label = []
                        for idx in trios_idx_mean_sorted:
                            # org image will be chosen
                            to_label.append(pool_trios[3*idx])
                        # original and all augmentations will be labled
                        active_set.label(to_label[: hyperparams["query_size"]])
                    else:
                        break
                else: 
                    break
            elif ((strategy == "mean_pandas") or (strategy == "variance_pandas")):
            
                pool = active_set.pool
                if len(pool) > 0:
                    probs = model.predict_on_dataset(
                        pool,
                        batch_size=hyperparams["batch_size"],
                        iterations=hyperparams["iterations"],
                        use_cuda=use_cuda,
                    )

                    if probs is not None and len(probs) > 0:
                        # 1. Get uncertainty
                        uncertainty = heuristic.get_uncertainties(probs)
                        oracle_indices = np.argsort(uncertainty)

                        if (hyperparams["augment"] != 1) and (hyperparams["augment"] != 2):
                            print("WARNING! Supporting only augmentation 1 and 2, for more write more code!")
                            sys.exit()
                        if hyperparams["augment"] == 1:
                            orig_s2 = int((len(pool)/2)-1)
                            aug1_s1 = int(len(pool)/2)
                            aug1_s2 = int((len(pool)/2)*2-1)

                            original = uncertainty[0:orig_s2]
                            aug1 = uncertainty[aug1_s1:aug1_s2]

                            if len(original) != len(aug1):
                                # at least one list has a different length (take shorter and fill with 0 to match arrays equel length)
                                if len(original) > len(aug1):
                                    aug1 += (len(original)-len(aug1)) * [0]
                                else:
                                    original += (len(aug1)-len(original)) * [0]

                            matrix = np.vstack([original, aug1])
                        if hyperparams["augment"] == 2: 
                            orig_s2 = int((len(pool)/3)-1)
                            aug1_s1 = int(len(pool)/3)
                            aug1_s2 = int((len(pool)/3)*2-1)
                            aug2_s1 = int((len(pool)/3)*2)
                            aug2_s2 = int(len(pool)-1)

                            original = uncertainty[0:orig_s2]
                            aug1 = uncertainty[aug1_s1:aug1_s2]
                            aug2 = uncertainty[aug2_s1:aug2_s2] 
                            print("3 original length "+str(len(original)))
                            print("4 aug1 length "+str(len(aug1)))
                            print("5 aug2 length "+str(len(aug2)))

                            if len(original) != len(aug1) or len(original) != len(aug2) or len(aug1) != len(aug2):
                                # at least one list has a different length (take shorter and fill with 0 to match arrays equel length)
                                if len(original) > len(aug1):
                                    aug1 += (len(original)-len(aug1)) * [0]
                                else:
                                    original += (len(aug1)-len(original)) * [0]
                                if len(original) > len(aug2):
                                    aug2 += (len(original)-len(aug2)) * [0]
                                else:
                                    original += (len(aug2)-len(original)) * [0]
                                if len(aug1) > len(aug2):
                                    aug2 += (len(aug1)-len(aug2)) * [0]
                                else:
                                    aug1 += (len(aug2)-len(aug1)) * [0]
                                
                            matrix = np.vstack([original, aug1, aug2])

                        # 2. Calc standard deviation
                        df_lab_img = pd.DataFrame(matrix)
                        if (strategy == "mean_pandas"):
                            mean_array = df_lab_img.mean()
                        else: 
                            mean_array = df_lab_img.std()
                        df_lab_img = pd.DataFrame(np.vstack([matrix, mean_array]))
                        
                        # 3. Map std uncertainties to uncertainty array
                        if hyperparams["augment"] == 1:
                            for i in range(len(uncertainty)):
                                uncertainty[i] = mean_array[i % (len(pool)/2-1)]
                        if hyperparams["augment"] == 2:
                            for i in range(len(uncertainty)):
                                uncertainty[i] = mean_array[i % (len(pool)/3-1)]
                        oracle_indices = np.argsort(uncertainty) # aufsteigend
                        to_label = oracle_indices[::-1] # absteigend
                        if len(to_label) > 0:
                            active_set.label(to_label[: hyperparams.get("query_size", 1)])
                        else: break
                    else:
                        break
                else: 
                    break

            # replacement for step
            
            n_augmented_old = active_set.n_augmented_images_labelled
            n_original_old = active_set.n_unaugmented_images_labelled
            
            test_acc = metrics["test_accuracy"].value
            train_acc = metrics["train_accuracy"].value
            test_loss = metrics["test_loss"].value
            train_loss = metrics["train_loss"].value 


            logs = {
                "epoch": epoch,
                "test_acc": test_acc,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "train_loss": train_loss,
                "Next training size": active_set.n_labelled,
                "new original images labelled": active_set.n_unaugmented_images_labelled - n_original_old,
                "new augmented images labelled": active_set.n_augmented_images_labelled - n_augmented_old
            }
            pprint(logs)

            csvwriter.writerow(
                (
                    epoch,
                    test_acc,
                    train_acc,
                    test_loss,
                    train_loss,
                    active_set.n_labelled,
                    active_set.n_unaugmented_images_labelled,
                    active_set.n_augmented_images_labelled
                )
            )

            tensorboardwriter.add_scalar("loss/train", train_loss, epoch)
            tensorboardwriter.add_scalar("loss/test", test_loss, epoch)
            tensorboardwriter.add_scalar("accuracy/train", train_acc, epoch)
            tensorboardwriter.add_scalar("accuracy/test",test_acc, epoch)
        torch.save(model, chkp_path)
        tensorboardwriter.close()
    

if __name__ == "__main__":
    main()
