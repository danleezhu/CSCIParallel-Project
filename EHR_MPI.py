## mpiexec -n 5 python EHR_MPI.py

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as ss
# !pip install imblearn
import imblearn
from imblearn.metrics import specificity_score

from sklearn.model_selection import cross_validate
import time
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    roc_auc_score
)
from imblearn.metrics import specificity_score
from scipy.stats import multivariate_normal

import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import svm

import warnings
warnings.filterwarnings('ignore')

from mpi4py import MPI
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time


def plot_confusion(y,y_pred):
    plt.rc('xtick', labelsize=10)  
    plt.rc('ytick', labelsize=10)
    c_m = confusion_matrix(y, y_pred)
    cm_display = ConfusionMatrixDisplay(confusion_matrix = c_m, display_labels = ['Positive', 'Negative'])
    cm_display.plot()
    plt.title("Model")
    plt.show()

def predict_score(model,y,ypre):
    np.random.seed(22)
    accuray = accuracy_score(y, ypre)
    f1 = f1_score(y, ypre,average="weighted")
    precision = precision_score(y, ypre,average="weighted")
    recall = recall_score(y, ypre,average="weighted")
    specificity = specificity_score(y, ypre,average="weighted")
    print("Predict on x (Accuray): ", accuray.round(4))
    print("Predict on x (Precision): ",precision.round(4))
    print("Predict on x (Recall): ", recall.round(4))
    print("Predict on x (specificity): ", specificity.round(4))
    print("Predict on x (f1-score): ", f1.round(4))
    
    plot_confusion(y,ypre)

def logistic_regression(comm, rank, size):
    Train = None
    Test = None
    chunk_size = None
    displacements = None
    local_data = None
    if rank == 0:
        print("************Logitstic***********")
        data = pd.read_csv("balancData.csv")
        split_rate = 0.8
        data = data.sample(frac=1, random_state=1)
        Train = data.iloc[:int(data.shape[0] * split_rate), :]
        Test = data.iloc[int(data.shape[0] * split_rate):, :]
        Test = Test.to_numpy()
        chunk_size = Test.shape[0] // size
        remainder = Test.shape[0] % size
        # Calculate the displacement for each rank
        displacements = [i * chunk_size + min(i, remainder) for i in range(size)]        
        # Calculate the size for each rank
        recv_size = chunk_size + (1 if rank < remainder else 0)
        # Scatter the data to different ranks
        local_data = np.empty((recv_size,18), dtype=int)

    Train = comm.bcast(Train, root=0)
    chunk_size = comm.bcast(chunk_size, root=0)
    local_data = np.zeros((chunk_size,18))
    comm.Scatterv([Test, (chunk_size,) * size, displacements, MPI.INT], local_data, root = 0)
    
    y_test = local_data[:,-1]
    X_test = local_data[:,0:-1]

    # Perform logistic regression
    
    logreg = LogisticRegression(random_state = 1)
    y_train_sample = Train['label']
    X_train_sample = Train.drop(['label'], axis=1)
    t = time.time()
    model = logreg.fit(X_train_sample, y_train_sample)
    print("Logistic regression training time   Rank", rank,": ", time.time() - t)
    start1 = time.time()
    y_test_pred = model.predict(X_test)
    print("Logistic regression predicting time Rank", rank,": ", time.time() - start1)
    
    # Gather predicted probabilities from all processes
    y_test_pred = comm.gather(y_test_pred, root=0)
    
    # if rank == 0:
    #   predict_score(model,y_test,y_test_pred)
       
def svmmodel(comm, rank, size):
    Train = None
    Test = None
    chunk_size = None
    displacements = None
    local_data = None
    if rank == 0:
        print("****************SVM***************")
        
        data = pd.read_csv("balancData.csv")
        split_rate = 0.8
        data = data.sample(frac=1, random_state=1)
        Train = data.iloc[:int(data.shape[0] * split_rate), :]
        Test = data.iloc[int(data.shape[0] * split_rate):, :]
        Test = Test.to_numpy()
        chunk_size = Test.shape[0] // size
        remainder = Test.shape[0] % size
        # Calculate the displacement for each rank
        displacements = [i * chunk_size + min(i, remainder) for i in range(size)]        
        # Calculate the size for each rank
        recv_size = chunk_size + (1 if rank < remainder else 0)
        # Scatter the data to different ranks
        local_data = np.empty((recv_size,18), dtype=int)

    Train = comm.bcast(Train, root=0)
    chunk_size = comm.bcast(chunk_size, root=0)
    local_data = np.zeros((chunk_size,18))
    comm.Scatterv([Test, (chunk_size,) * size, displacements, MPI.INT], local_data, root = 0)
    
    y_test = local_data[:,-1]
    X_test = local_data[:,0:-1]

    # Perform logistic regression
    svmm = svm.LinearSVC(random_state = 1)
    y_train_sample = Train['label']
    X_train_sample = Train.drop(['label'], axis=1)
    t1 = time.time()
    model = svmm.fit(X_train_sample, y_train_sample)
    print("SVM training time   Rank", rank,": ", time.time() - t1)
    start1 = time.time()
    y_test_pred = model.predict(X_test)
    print("SVM predicting time Rank", rank,": ", time.time() - start1)
    
    # Gather predicted probabilities from all processes
    y_test_pred = comm.gather(y_test_pred, root=0)

    # if rank == 0:
    #     predict_score(model,y_test, y_test_pred)
       
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    logistic_regression(comm, rank, size)
    svmmodel(comm, rank, size)
    





    