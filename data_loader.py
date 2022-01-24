import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

class Loader:
    def __init__(self,type=0,test=False,method='minmax'):
        self.train = pd.read_csv('data/all_dataset_train.csv')
        #self.train, self.test = train_test_split(self.dataset, test_size=0.3)
        model = SMOTE(k_neighbors=4,sampling_strategy = 1)
        self.x_train = self.train.drop(['target'],axis=1)
        self.y_train = self.train['target']
        self.balanced_x_train,self.balanced_y_train = model.fit_sample(self.x_train,self.y_train)
        self.type=type
        self.method = method
        self.actual_x_train = None
        self.actual_x_test = None
        self.process_data()
        if(test):
            self.test = pd.read_csv('data/all_dataset_test.csv')
            self.x_test = self.test.drop(['target'],axis=1)
            self.y_test = self.test['target']
            self.balanced_x_test,self.balanced_y_test = model.fit_sample(self.x_test,self.y_test)
            self.process_test_data()

    def process_data(self):
        #print("iniciando procesamiento")
        if self.type==0:
            #self.actual_x_train = self.balanced_x_train.drop(["Kurtosis2","Skewness2","Autocorrelation2","Entropy2","WL2","Kurtosis3","Skewness3","Autocorrelation3","Entropy3","WL3"],axis=1)
            self.actual_x_train = self.balanced_x_train[["l_Maximum3", "l_Skewness", "l_Maximum2", "l_A_Coefficient3", "l_Minimum2"]]
            #print("este es x_train 0")
        elif self.type==1:
            self.actual_x_train = self.balanced_x_train[["l_Maximum3", "l_Skewness", "l_Maximum2", "l_A_Coefficient3", "l_Kurtosis"]]
            #print("este es x_train 1")
        elif self.type==2:
            self.actual_x_train = self.balanced_x_train[["l_Maximum3", "l_Skewness", "l_Maximum2", "l_A_Coefficient3", "l_B_Coefficient3"]]
            #print("este es x_train 2")
        elif self.type==3:
            self.actual_x_train = self.balanced_x_train
            #print("este es x_train completo")
        #print("este es x_train en process data")
        if self.method == 'minmax':
            scaler = MinMaxScaler()
        elif self.method == 'standard':
            scaler = StandardScaler()
        self.actual_x_train = scaler.fit_transform(self.actual_x_train)
        self.actual_x_train = self.actual_x_train*10

    def process_test_data(self):
        if self.type==0:
            self.actual_x_test = self.balanced_x_test[["l_Maximum3", "l_Skewness", "l_Maximum2", "l_A_Coefficient3", "l_Minimum2"]]
            #print("este es x_test 0")
        elif self.type==1:
            self.actual_x_test = self.balanced_x_test[["l_Maximum3", "l_Skewness", "l_Maximum2", "l_A_Coefficient3", "l_Kurtosis"]]
            #print("este es x_test 1")
        elif self.type==2:
            self.actual_x_test = self.balanced_x_test[["l_Maximum3", "l_Skewness", "l_Maximum2", "l_A_Coefficient3", "l_B_Coefficient3"]]
            #print("este es x_test 2")
        elif self.type==3:
            self.actual_x_test = self.balanced_x_test
            #print("este es x_test completo")
        #print("este es x_test en process data")
        if self.method == 'minmax':
            scaler = MinMaxScaler()
        elif self.method == 'standard':
            scaler = StandardScaler()
        self.actual_x_test = scaler.fit_transform(self.actual_x_test)
        self.actual_x_test = self.actual_x_test*10

    def get_train_data(self):
        return self.actual_x_train,self.balanced_y_train
    
    def get_test_data(self):
        return self.actual_x_test,self.balanced_y_test
