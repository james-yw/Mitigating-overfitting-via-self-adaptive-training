import os

import numpy as np
import matplotlib.pyplot as plt


def plot():
#     sat_result_dir = "./results/SVHN/resnet18_sat_corrupted_label_r0_cosine_"
    sat_result_dir = "./results/SVHN/resnet18_sat_corrupted_label_r0.4_cosine_"
#     sat_result_dir = "./results/SVHN/resnet18_sat_shuffled_pixels_r0.4_cosine_"
#     sat_result_dir = "./results/SVHN/resnet18_sat_random_pixels_r0.4_cosine_"
#     sat_result_dir = "./results/SVHN/resnet18_sat_Gaussian_r0.4_cosine_"
    SAT = {}
    
#     ce_result_dir = "./results/SVHN/resnet18_ce_corrupted_label_r0_cosine_"
    ce_result_dir = "./results/SVHN/resnet18_ce_corrupted_label_r0.4_cosine_"
#     ce_result_dir = "./results/SVHN/resnet18_ce_shuffled_pixels_r0.4_cosine_"
#     ce_result_dir = "./results/SVHN/resnet18_ce_random_pixels_r0.4_cosine_"
#     ce_result_dir = "./results/SVHN/resnet18_ce_Gaussian_r0.4_cosine_"
    
    CE = {}


    SAT['train_loss'] = np.load(os.path.join(sat_result_dir,"train_loss.npy"))
    SAT['train_acc'] = np.load(os.path.join(sat_result_dir, "train_acc.npy"))
    SAT['test_loss'] = np.load(os.path.join(sat_result_dir, "test_loss.npy"))
    SAT['test_acc'] = np.load(os.path.join(sat_result_dir, "test_acc.npy"))

    CE['train_loss'] = np.load(os.path.join(ce_result_dir, "train_loss.npy"))
    CE['train_acc'] = np.load(os.path.join(ce_result_dir, "train_acc.npy"))
    CE['test_loss'] = np.load(os.path.join(ce_result_dir, "test_loss.npy"))
    CE['test_acc'] = np.load(os.path.join(ce_result_dir, "test_acc.npy"))
    
    print("SAT test loss:",np.argmin(SAT['test_loss']))
    print("CE test loss:",np.argmin(CE['test_loss']))

    plt.figure(figsize=(10,8))
    
    
    plt.subplot(2,2,1)
    plt.title("train_loss")
    plt.plot(SAT['train_loss'], color='red', label='SAT_train_loss')
    plt.plot(CE['train_loss'], color='blue', label='CE_train_loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.title("train_acc")
    plt.plot(SAT['train_acc'], color='red', label='SAT_train_acc')
    plt.plot(CE['train_acc'], color='blue', label='CE_train_acc')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.title("test_loss")
    plt.plot(SAT['test_loss'], color='red', label='SAT_test_loss')
    plt.plot(CE['test_loss'], color='blue', label='CE_test_loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.title("test_acc")
    plt.plot(SAT['test_acc'], color='red', label='SAT_test_acc')
    plt.plot(CE['test_acc'], color='blue', label='CE_test_acc')
    plt.legend()

    plt.show()
    #plt.savefig("./result_fig/result_shuffled_pixels_r0.4.jpg")
    
if __name__=="__main__":
    plot()


