import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from data_loader import Loader
from scipy.optimize import minimize

def initialize_fuzzy(type):
    global new
    new = Loader(type)

def generate_fuzzy(x):
    kurtosis = ctrl.Antecedent(np.arange(0, 11, 1), 'kurtosis')
    skew = ctrl.Antecedent(np.arange(0, 11, 1), 'skewness')
    autocorr = ctrl.Antecedent(np.arange(0, 11, 1), 'autocorrelation')
    entropy = ctrl.Antecedent(np.arange(0, 11, 1), 'entropy')
    wl = ctrl.Antecedent(np.arange(0, 11, 1), 'wavelet length')
    transit = ctrl.Consequent(np.arange(0, 11, 1), 'transit')

    k_p = kurtosis['p'] = fuzz.gaussmf(kurtosis.universe, x[0], x[1])
    k_n = kurtosis['np'] = fuzz.gaussmf(kurtosis.universe, x[2], x[3])
    s_p = skew['p'] = fuzz.gaussmf(skew.universe, x[4], x[5])
    s_n = skew['np'] = fuzz.gaussmf(skew.universe, x[6], x[7])
    a_p = autocorr['p'] = fuzz.gaussmf(autocorr.universe, x[8], x[9])
    a_n = autocorr['np'] = fuzz.gaussmf(autocorr.universe, x[10], x[11])
    e_p = entropy['p'] = fuzz.gaussmf(entropy.universe, x[12], x[13])
    e_n = entropy['np'] = fuzz.gaussmf(entropy.universe, x[14], x[15])
    w_p = wl['p'] = fuzz.gaussmf(wl.universe, x[16], x[17])
    w_n = wl['np'] = fuzz.gaussmf(wl.universe, x[18], x[19])
    t_p = transit['p'] = fuzz.gaussmf(transit.universe, x[20], x[21])
    t_n = transit['np'] = fuzz.gaussmf(transit.universe, x[22], x[23])

    rules_t = ctrl.Rule(antecedent=((kurtosis['p'] & skew['p'] & autocorr['p'] & entropy['p'] & wl['p']) |
                              (kurtosis['p'] & skew['p'] & autocorr['p'] & entropy['np'] & wl['p']) |
                              (kurtosis['p'] & skew['p'] & autocorr['np'] & entropy['np'] & wl['p']) |
                              (kurtosis['p'] & skew['np'] & autocorr['p'] & entropy['np'] & wl['p']) |
                              (kurtosis['np'] & skew['p'] & autocorr['p'] & entropy['np'] & wl['p']) |
                              (kurtosis['p'] & skew['p'] & autocorr['np'] & entropy['p'] & wl['p']) |
                              (kurtosis['p'] & skew['np'] & autocorr['np'] & entropy['p'] & wl['p']) |
                              (kurtosis['np'] & skew['p'] & autocorr['np'] & entropy['p'] & wl['p']) |
                              (kurtosis['p'] & skew['np'] & autocorr['p'] & entropy['p'] & wl['p']) |
                              (kurtosis['np'] & skew['np'] & autocorr['p'] & entropy['p'] & wl['p']) |
                              (kurtosis['np'] & skew['p'] & autocorr['p'] & entropy['p'] & wl['p']) |
                              (kurtosis['p'] & skew['p'] & autocorr['p'] & entropy['p'] & wl['np']) |
                              (kurtosis['p'] & skew['p'] & autocorr['p'] & entropy['np'] & wl['np']) |
                              (kurtosis['np'] & skew['p'] & autocorr['p'] & entropy['np'] & wl['np']) |
                              (kurtosis['p'] & skew['p'] & autocorr['np'] & entropy['p'] & wl['np']) |
                              (kurtosis['p'] & skew['np'] & autocorr['p'] & entropy['p'] & wl['np']) |
                              (kurtosis['np'] & skew['p'] & autocorr['p'] & entropy['p'] & wl['np'])
                              ),
                        consequent=transit['p'], label='rule transit')

    rules_nt = ctrl.Rule(antecedent=((kurtosis['np'] & skew['np'] & autocorr['np'] & entropy['np'] & wl['p']) |
                              (kurtosis['p'] & skew['np'] & autocorr['np'] & entropy['np'] & wl['p']) |
                              (kurtosis['np'] & skew['p'] & autocorr['np'] & entropy['np'] & wl['p']) |
                              (kurtosis['np'] & skew['np'] & autocorr['p'] & entropy['np'] & wl['p']) |
                              (kurtosis['np'] & skew['np'] & autocorr['np'] & entropy['p'] & wl['p']) | 
                              (kurtosis['np'] & skew['np'] & autocorr['np'] & entropy['np'] & wl['np']) |
                              (kurtosis['p'] & skew['np'] & autocorr['np'] & entropy['np'] & wl['np']) |
                              (kurtosis['np'] & skew['p'] & autocorr['np'] & entropy['np'] & wl['np']) |
                              (kurtosis['np'] & skew['np'] & autocorr['p'] & entropy['np'] & wl['np']) |
                              (kurtosis['np'] & skew['np'] & autocorr['np'] & entropy['p'] & wl['np'])
                              ),
                        consequent=transit['np'], label='rule not transit')
    
    # Comentar al momento de optimizar 
    """ 
    fig, (ax0, ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=6, figsize=(8, 15))

    ax0.plot(kurtosis.universe, k_p, 'g', linewidth=1.5, label='Planeta')
    ax0.plot(kurtosis.universe, k_n, 'r', linewidth=1.5, label='No planeta')
    ax0.set_title('Kurtosis')
    ax0.legend()

    ax1.plot(skew.universe, s_p, 'g', linewidth=1.5, label='Planeta')
    ax1.plot(skew.universe, s_n, 'r', linewidth=1.5, label='No planeta')
    ax1.set_title('Skew')
    ax1.legend()

    ax2.plot(autocorr.universe, a_p, 'g', linewidth=1.5, label='Planeta')
    ax2.plot(autocorr.universe, a_n, 'r', linewidth=1.5, label='No planeta')
    ax2.set_title('Autocorrelation')
    ax2.legend()

    ax3.plot(entropy.universe, e_p, 'g', linewidth=1.5, label='Planeta')
    ax3.plot(entropy.universe, e_n, 'r', linewidth=1.5, label='No planeta')
    ax3.set_title('Entropy')
    ax3.legend()

    ax4.plot(wl.universe, w_p, 'g', linewidth=1.5, label='Planeta')
    ax4.plot(wl.universe, w_n, 'r', linewidth=1.5, label='No planeta')
    ax4.set_title('Wavelet length')
    ax4.legend()

    ax5.plot(transit.universe, t_p, 'g', linewidth=1.5, label='Planeta')
    ax5.plot(transit.universe, t_n, 'r', linewidth=1.5, label='No planeta')
    ax5.set_title('Transit')
    ax5.legend()

    for ax in (ax0, ax1, ax2, ax3, ax4, ax5):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()  """
    # Comentar hasta aqui

    sys = ctrl.ControlSystem(rules=[rules_t, rules_nt])
    fis = ctrl.ControlSystemSimulation(sys)
    return fis

def evaluate_fuzzy(x):
    fis = generate_fuzzy(x)

    x_train,y_train = new.get_train_data()

    tp,tn,fp,fn = evaluation(x_train,y_train,fis)

    # Comentar print() al momento de optimizar
    """ print("Confusion Matrix (train data)")
    
    print("["+str(tp)+", "+str(fp)+"]")
    print("["+str(fn)+", "+str(tn)+"]") """
    
    accuracy = (tp + tn)/(tp+tn+fp+fn)
    e = 1 - accuracy
    
    #fsp = tp/(tp+0.5*(fp+fn))
    #fsnp = tn/(tn+0.5*(fp+fn))
    #f_score = (tp)/(tp+0.5*(fp+fn))

    # print("F-score (average)")
    # print(f_score)
    #e = 1-f_score

    # Comentar a la hora de optimizar
    """ x_test,y_test = new.get_test_data()

    tpt,tnt,fpt,fnt = evaluation(x_test,y_test,fis)

    print("Confusion Matrix (test data)")
    
    print("["+str(tpt)+", "+str(fpt)+"]")
    print("["+str(fnt)+", "+str(tnt)+"]") 
    
    fspt = tpt/(tpt+0.5*(fpt+fnt))
    fsnpt = tnt/(tnt+0.5*(fpt+fnt))
    f_scoret = (fspt+fsnpt)/2

    print("F-score (average)")
    print(f_scoret) """
    # Comentar hasta aquí

    return e

def evaluation(x,y,fis):
    n = len(y)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(n):
        x_in = x[i,:]
        fis.input['kurtosis']=x_in[0]
        fis.input['skewness']=x_in[1]
        fis.input['autocorrelation']=x_in[2]
        fis.input['entropy']=x_in[3]
        fis.input['wavelet length']=x_in[4]
        fis.compute()
        out = round(fis.output['transit']/10)
        if out == y[i] and y[i] == 1:
            tp += 1
        elif out == y[i] and y[i] == 0:
            tn += 1
        elif out != y[i] and y[i] == 1:
            fn += 1
        elif out != y[i] and y[i] == 0:
            fp += 1

    return tp,tn,fp,fn
