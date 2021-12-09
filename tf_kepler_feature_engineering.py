from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import scipy
from scipy.stats import kurtosis, skew, entropy, norm
from scipy import fftpack
import tensorflow as tf
import os.path
import warnings

warnings.simplefilter("ignore")

kepler_data_path = "/home/newpc1/Documentos/ExoplanetML/data/kepler"
kepler_out_data_path = "/home/newpc1/Documentos/ExoplanetML/data/kepler-processed-dataset"
labels_path = "/home/newpc1/Documentos/ExoplanetML/data/q1_q17_dr24_tce_labeled.csv"
labels_file = pd.read_csv(labels_path,skiprows=17)
TFRECORD_TRAIN_DIR = "/home/newpc1/Documentos/ExoplanetML/data/precomputed-dataset/kepler/tfrecord/train"
TFRECORD_TEST_DIR = "/home/newpc1/Documentos/ExoplanetML/data/precomputed-dataset/kepler/tfrecord/test"

# def find_tce(kepid, tce_plnt_num, filenames):
#     for filename in filenames:
#         for record in tf.python_io.tf_record_iterator(filename):
#             ex = tf.train.Example.FromString(record)
#             if (ex.features.feature["kepid"].int64_list.value[0] == kepid and
#                 ex.features.feature["tce_plnt_num"].int64_list.value[0] == tce_plnt_num):
#                 print("Found {}_{} in file {}".format(kepid, tce_plnt_num, filename))
#                 return ex
#     raise ValueError("{}_{} not found in files: {}".format(kepid, tce_plnt_num, filenames))

def find_tce(kepid, tce_plnt_num, filenames):
    for filename in filenames:
        #for record in tf.data.TFRecordDataset(filename):
        for record in tf.compat.v1.io.tf_record_iterator(filename):
            ex = tf.train.Example.FromString(record)
            if (ex.features.feature["kepid"].int64_list.value[0] == kepid and
                ex.features.feature["tce_plnt_num"].int64_list.value[0] == tce_plnt_num):
                print("Found {}_{} in file {}".format(kepid, tce_plnt_num, filename))
                return ex
    raise ValueError("{}_{} not found in files: {}".format(kepid, tce_plnt_num, filenames))
    
def plot_tce(kepid, tce_plnt_num):
    filenames = tf.io.gfile.glob(os.path.join(TFRECORD_TRAIN_DIR, "*"))
    assert filenames, "No files found in {}".format(TFRECORD_TRAIN_DIR)
    ex = find_tce(kepid, tce_plnt_num, filenames)
    global_view = np.array(ex.features.feature["global_view"].float_list.value)
    local_view = np.array(ex.features.feature["local_view"].float_list.value)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].plot(global_view, ".")
    axes[1].plot(local_view, ".")
    plt.show()
    
def plot_tce_grad(kepid, tce_plnt_num):
    filenames = tf.io.gfile.glob(os.path.join(TFRECORD_TRAIN_DIR, "*"))
    assert filenames, "No files found in {}".format(TFRECORD_TRAIN_DIR)
    ex = find_tce(kepid, tce_plnt_num, filenames)
    global_view = np.array(ex.features.feature["global_view"].float_list.value)
    local_view = np.array(ex.features.feature["local_view"].float_list.value)
    grad_global_view = get_gradient(global_view)
    grad_local_view = get_gradient(local_view)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].plot(grad_global_view, ".")
    axes[1].plot(grad_local_view, ".")
    plt.show()
    
def plot_tce_fft(kepid, tce_plnt_num):
    filenames = tf.io.gfile.glob(os.path.join(TFRECORD_TRAIN_DIR, "*"))
    assert filenames, "No files found in {}".format(TFRECORD_TRAIN_DIR)
    ex = find_tce(kepid, tce_plnt_num, filenames)
    global_view = np.array(ex.features.feature["global_view"].float_list.value)
    local_view = np.array(ex.features.feature["local_view"].float_list.value)
    fft_global_view = np.abs(scipy.fftpack.fft(global_view))
    fft_local_view = np.abs(scipy.fftpack.fft(local_view))
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))
    axes[0].plot(fft_global_view, ".")
    axes[1].plot(fft_local_view, ".")
    plt.show()

def get_wavelet_length(x):
    p = []
    for i in range(0,len(x)-1):
        p.append(np.abs(x[i+1]-x[i]))
    return sum(p)

def get_gradient(x):
    grad = []
    grad.append(0)
    for i in range(1,len(x)-1):
        conv = -1*x[i-1]+x[i+1]
        grad.append(conv)
    grad.append(0)
    return grad

def process_train_dataset():
    tce_kepids = []
    tce_plnt_nums = []
    gv_kurtosis = []
    gv_skewness = []
    gv_autocorrelation = []
    gv_entropy = []
    gv_wavelet_length = []
    gv_grad_kurtosis = []
    gv_grad_skewness = []
    gv_grad_autocorrelation = []
    gv_grad_entropy = []
    gv_grad_wavelet_length = []
    gv_fft_kurtosis = []
    gv_fft_skewness = []
    gv_fft_autocorrelation = []
    gv_fft_entropy = []
    gv_fft_wavelet_length = []
    lv_kurtosis = []
    lv_skewness = []
    lv_autocorrelation = []
    lv_entropy = []
    lv_wavelet_length = []
    lv_grad_kurtosis = []
    lv_grad_skewness = []
    lv_grad_autocorrelation = []
    lv_grad_entropy = []
    lv_grad_wavelet_length = []
    lv_fft_kurtosis = []
    lv_fft_skewness = []
    lv_fft_autocorrelation = []
    lv_fft_entropy = []
    lv_fft_wavelet_length = []
    labels = []
    filenames = tf.io.gfile.glob(os.path.join(TFRECORD_TRAIN_DIR, "*"))
    assert filenames, "No files found in {}".format(TFRECORD_TRAIN_DIR)
    print("Iniciando procesamiento de conjunto de datos de entrenamiento...")
    for filename in filenames:
        #for record in tf.data.TFRecordDataset(filename):
        print("Procesando archivo "+filename+"...")
        i = 0
        for record in tf.compat.v1.io.tf_record_iterator(filename):
            ex = tf.train.Example.FromString(record)
            if i%10==0 and i>0:
                print(str(i)+" records procesados")
            kepid = ex.features.feature["kepid"].int64_list.value[0]
            tce_plnt_num = ex.features.feature["tce_plnt_num"].int64_list.value[0]
            tce_kepids.append(kepid)
            tce_plnt_nums.append(tce_plnt_num)
            global_view = np.array(ex.features.feature["global_view"].float_list.value)
            local_view = np.array(ex.features.feature["local_view"].float_list.value)
            target = ex.features.feature["av_training_set"].bytes_list.value[0].decode()
            if target == 'PC':
                label = 1
            else:
                label = 0
            labels.append(label)
        #Global view feature extraction
            grad_global_view = get_gradient(global_view)
            fft_global_view = np.abs(scipy.fftpack.fft(global_view))
            #Normalized light curve
            gv_kurtosis.append(kurtosis(global_view))
            gv_skewness.append(skew(global_view))
            gv_autocorrelation.append(np.sum(np.correlate(global_view, global_view, mode="full")))
            gv_entropy.append(entropy((np.array(global_view)+1),base=2))
            gv_wavelet_length.append(get_wavelet_length(global_view))
            #Gradient 
            gv_grad_kurtosis.append(kurtosis(grad_global_view))
            gv_grad_skewness.append(skew(grad_global_view))
            gv_grad_autocorrelation.append(np.sum(np.correlate(grad_global_view, grad_global_view, mode="full")))
            gv_grad_entropy.append(entropy((np.array(grad_global_view)+1),base=2))
            gv_grad_wavelet_length.append(get_wavelet_length(grad_global_view))
            #Fourier transform
            gv_fft_kurtosis.append(kurtosis(fft_global_view))
            gv_fft_skewness.append(skew(fft_global_view))
            gv_fft_autocorrelation.append(np.sum(np.correlate(fft_global_view, fft_global_view, mode="full")))
            gv_fft_entropy.append(entropy((np.array(fft_global_view)+1),base=2))
            gv_fft_wavelet_length.append(get_wavelet_length(fft_global_view))
        #Local view feature extraction
            grad_local_view = get_gradient(local_view)
            fft_local_view = np.abs(scipy.fftpack.fft(local_view))
            #Normalized light curve
            lv_kurtosis.append(kurtosis(local_view))
            lv_skewness.append(skew(local_view))
            lv_autocorrelation.append(np.sum(np.correlate(local_view, local_view, mode="full")))
            lv_entropy.append(entropy((np.array(local_view)+1),base=2))
            lv_wavelet_length.append(get_wavelet_length(local_view))
            #Gradient 
            lv_grad_kurtosis.append(kurtosis(grad_local_view))
            lv_grad_skewness.append(skew(grad_local_view))
            lv_grad_autocorrelation.append(np.sum(np.correlate(grad_local_view, grad_local_view, mode="full")))
            lv_grad_entropy.append(entropy((np.array(grad_local_view)+1),base=2))
            lv_grad_wavelet_length.append(get_wavelet_length(grad_local_view))
            #Fourier transform
            lv_fft_kurtosis.append(kurtosis(fft_local_view))
            lv_fft_skewness.append(skew(fft_local_view))
            lv_fft_autocorrelation.append(np.sum(np.correlate(fft_local_view, fft_local_view, mode="full")))
            lv_fft_entropy.append(entropy((np.array(fft_local_view)+1),base=2))
            lv_fft_wavelet_length.append(get_wavelet_length(fft_local_view))
            
            i=i+1
        print(str(i)+" records procesados en archivo "+filename+". Inicia procesamiento de siguiente archivo.")
            
    dict_gv_dataset = {"tce_kepid:": tce_kepids,
                        "tce_plnt_num:": tce_plnt_nums,
                        
                        "Kurtosis": gv_kurtosis, 
                        "Skewness": gv_skewness, 
                        "Autocorrelation": gv_autocorrelation, 
                        "Entropy": gv_entropy,
                        "WL":gv_wavelet_length,

                        "Kurtosis2": gv_grad_kurtosis, 
                        "Skewness2": gv_grad_skewness, 
                        "Autocorrelation2": gv_grad_autocorrelation, 
                        "Entropy2": gv_grad_entropy, 
                        "WL2":gv_grad_wavelet_length,

                        "Kurtosis3": gv_fft_kurtosis, 
                        "Skewness3": gv_fft_skewness, 
                        "Autocorrelation3": gv_fft_autocorrelation, 
                        "Entropy3": gv_fft_entropy, 
                        "WL3":gv_fft_wavelet_length,

                        "target":labels}

    dict_lv_dataset = {"tce_kepid:": tce_kepids,
                        "tce_plnt_num:": tce_plnt_nums,
                        
                        "Kurtosis": lv_kurtosis, 
                        "Skewness": lv_skewness, 
                        "Autocorrelation": lv_autocorrelation, 
                        "Entropy": lv_entropy,
                        "WL":lv_wavelet_length,

                        "Kurtosis2": lv_grad_kurtosis, 
                        "Skewness2": lv_grad_skewness, 
                        "Autocorrelation2": lv_grad_autocorrelation, 
                        "Entropy2": lv_grad_entropy, 
                        "WL2":lv_grad_wavelet_length,

                        "Kurtosis3": lv_fft_kurtosis, 
                        "Skewness3": lv_fft_skewness, 
                        "Autocorrelation3": lv_fft_autocorrelation, 
                        "Entropy3": lv_fft_entropy, 
                        "WL3":lv_fft_wavelet_length,

                        "target":labels}

    print("Todos los archivos han sido procesados, generando archivos CSV...")   
    df_gv_dataset = pd.DataFrame(dict_gv_dataset)
    df_gv_dataset = df_gv_dataset.sort_values(by='target',ascending=False)
    df_gv_dataset.to_csv(kepler_out_data_path+"/global_view_dataset_train.csv", sep=';')
    
    df_lv_dataset = pd.DataFrame(dict_lv_dataset)
    df_lv_dataset = df_lv_dataset.sort_values(by='target',ascending=False)
    df_lv_dataset.to_csv(kepler_out_data_path+"/local_view_dataset_train.csv", sep=';')
    print("Procesamiento de datos de entrenamiento finalizado.")   
            
def process_test_dataset():
    tce_kepids = []
    tce_plnt_nums = []
    gv_kurtosis = []
    gv_skewness = []
    gv_autocorrelation = []
    gv_entropy = []
    gv_wavelet_length = []
    gv_grad_kurtosis = []
    gv_grad_skewness = []
    gv_grad_autocorrelation = []
    gv_grad_entropy = []
    gv_grad_wavelet_length = []
    gv_fft_kurtosis = []
    gv_fft_skewness = []
    gv_fft_autocorrelation = []
    gv_fft_entropy = []
    gv_fft_wavelet_length = []
    lv_kurtosis = []
    lv_skewness = []
    lv_autocorrelation = []
    lv_entropy = []
    lv_wavelet_length = []
    lv_grad_kurtosis = []
    lv_grad_skewness = []
    lv_grad_autocorrelation = []
    lv_grad_entropy = []
    lv_grad_wavelet_length = []
    lv_fft_kurtosis = []
    lv_fft_skewness = []
    lv_fft_autocorrelation = []
    lv_fft_entropy = []
    lv_fft_wavelet_length = []
    labels = []
    filenames = tf.io.gfile.glob(os.path.join(TFRECORD_TEST_DIR, "*"))
    assert filenames, "No files found in {}".format(TFRECORD_TEST_DIR)
    print("Iniciando procesamiento de conjunto de datos de prueba...")
    for filename in filenames:
        #for record in tf.data.TFRecordDataset(filename):
        print("Procesando archivo "+filename+"...")
        i = 0
        for record in tf.compat.v1.io.tf_record_iterator(filename):
            ex = tf.train.Example.FromString(record)
            if i%10==0 and i>0:
                print(str(i)+" records procesados")
            kepid = ex.features.feature["kepid"].int64_list.value[0]
            tce_plnt_num = ex.features.feature["tce_plnt_num"].int64_list.value[0]
            tce_kepids.append(kepid)
            tce_plnt_nums.append(tce_plnt_num)
            global_view = np.array(ex.features.feature["global_view"].float_list.value)
            local_view = np.array(ex.features.feature["local_view"].float_list.value)
            target = ex.features.feature["av_training_set"].bytes_list.value[0].decode()
            if target == 'PC':
                label = 1
            else:
                label = 0
            labels.append(label)
        #Global view feature extraction
            grad_global_view = get_gradient(global_view)
            fft_global_view = np.abs(scipy.fftpack.fft(global_view))
            #Normalized light curve
            gv_kurtosis.append(kurtosis(global_view))
            gv_skewness.append(skew(global_view))
            gv_autocorrelation.append(np.sum(np.correlate(global_view, global_view, mode="full")))
            gv_entropy.append(entropy((np.array(global_view)+1),base=2))
            gv_wavelet_length.append(get_wavelet_length(global_view))
            #Gradient 
            gv_grad_kurtosis.append(kurtosis(grad_global_view))
            gv_grad_skewness.append(skew(grad_global_view))
            gv_grad_autocorrelation.append(np.sum(np.correlate(grad_global_view, grad_global_view, mode="full")))
            gv_grad_entropy.append(entropy((np.array(grad_global_view)+1),base=2))
            gv_grad_wavelet_length.append(get_wavelet_length(grad_global_view))
            #Fourier transform
            gv_fft_kurtosis.append(kurtosis(fft_global_view))
            gv_fft_skewness.append(skew(fft_global_view))
            gv_fft_autocorrelation.append(np.sum(np.correlate(fft_global_view, fft_global_view, mode="full")))
            gv_fft_entropy.append(entropy((np.array(fft_global_view)+1),base=2))
            gv_fft_wavelet_length.append(get_wavelet_length(fft_global_view))
        #Local view feature extraction
            grad_local_view = get_gradient(local_view)
            fft_local_view = np.abs(scipy.fftpack.fft(local_view))
            #Normalized light curve
            lv_kurtosis.append(kurtosis(local_view))
            lv_skewness.append(skew(local_view))
            lv_autocorrelation.append(np.sum(np.correlate(local_view, local_view, mode="full")))
            lv_entropy.append(entropy((np.array(local_view)+1),base=2))
            lv_wavelet_length.append(get_wavelet_length(local_view))
            #Gradient 
            lv_grad_kurtosis.append(kurtosis(grad_local_view))
            lv_grad_skewness.append(skew(grad_local_view))
            lv_grad_autocorrelation.append(np.sum(np.correlate(grad_local_view, grad_local_view, mode="full")))
            lv_grad_entropy.append(entropy((np.array(grad_local_view)+1),base=2))
            lv_grad_wavelet_length.append(get_wavelet_length(grad_local_view))
            #Fourier transform
            lv_fft_kurtosis.append(kurtosis(fft_local_view))
            lv_fft_skewness.append(skew(fft_local_view))
            lv_fft_autocorrelation.append(np.sum(np.correlate(fft_local_view, fft_local_view, mode="full")))
            lv_fft_entropy.append(entropy((np.array(fft_local_view)+1),base=2))
            lv_fft_wavelet_length.append(get_wavelet_length(fft_local_view))
            
            i=i+1
        print(str(i)+" records procesados en archivo "+filename+". Inicia procesamiento de siguiente archivo.")
            
    dict_gv_dataset = {"tce_kepid:": tce_kepids,
                        "tce_plnt_num:": tce_plnt_nums,
                        
                        "Kurtosis": gv_kurtosis, 
                        "Skewness": gv_skewness, 
                        "Autocorrelation": gv_autocorrelation, 
                        "Entropy": gv_entropy,
                        "WL":gv_wavelet_length,

                        "Kurtosis2": gv_grad_kurtosis, 
                        "Skewness2": gv_grad_skewness, 
                        "Autocorrelation2": gv_grad_autocorrelation, 
                        "Entropy2": gv_grad_entropy, 
                        "WL2":gv_grad_wavelet_length,

                        "Kurtosis3": gv_fft_kurtosis, 

                        "Skewness3": gv_fft_skewness, 
                        "Autocorrelation3": gv_fft_autocorrelation, 
                        "Entropy3": gv_fft_entropy, 
                        "WL3":gv_fft_wavelet_length,

                        "target":labels}

    dict_lv_dataset = {"tce_kepid:": tce_kepids,
                        "tce_plnt_num:": tce_plnt_nums,
                        
                        "Kurtosis": lv_kurtosis, 
                        "Skewness": lv_skewness, 
                        "Autocorrelation": lv_autocorrelation, 
                        "Entropy": lv_entropy,
                        "WL":lv_wavelet_length,

                        "Kurtosis2": lv_grad_kurtosis, 
                        "Skewness2": lv_grad_skewness, 
                        "Autocorrelation2": lv_grad_autocorrelation, 
                        "Entropy2": lv_grad_entropy, 
                        "WL2":lv_grad_wavelet_length,

                        "Kurtosis3": lv_fft_kurtosis, 
                        "Skewness3": lv_fft_skewness, 
                        "Autocorrelation3": lv_fft_autocorrelation, 
                        "Entropy3": lv_fft_entropy, 
                        "WL3":lv_fft_wavelet_length,

                        "target":labels}
    
    print("Todos los archivos han sido procesados, generando archivos CSV...")   
    df_gv_dataset = pd.DataFrame(dict_gv_dataset)
    df_gv_dataset = df_gv_dataset.sort_values(by='target',ascending=False)
    df_gv_dataset.to_csv(kepler_out_data_path+"/global_view_dataset_test.csv", sep=';')
    
    df_lv_dataset = pd.DataFrame(dict_lv_dataset)
    df_lv_dataset = df_lv_dataset.sort_values(by='target',ascending=False)
    df_lv_dataset.to_csv(kepler_out_data_path+"/local_view_dataset_test.csv", sep=';')
    print("Procesamiento de datos de prueba finalizado.") 
    
def create_extracted_datasets():
    process_train_dataset()
    process_test_dataset()


def plot_kepler_tce(kepid):
    kepid_str = str(kepid)
    digits = len(kepid_str)
    if(digits==6):
        zone = "000"+kepid_str[0]
        kepid_str = "000"+kepid_str
    elif(digits==7):
        zone = "00"+kepid_str[0]+kepid_str[1]
        kepid_str = "00"+kepid_str
    elif(digits==8):
        zone = "0"+kepid_str[0]+kepid_str[1]+kepid_str[2]
        kepid_str = "0"+kepid_str
    elif(digits==9):
        zone = kepid_str[0]+kepid_str[1]+kepid_str[2]+kepid_str[3]
    tce_fits_files = glob.glob(kepler_data_path+"/"+zone+"/"+kepid_str+"/*")
    time_of_tces = labels_file['tce_time0'].loc[labels_file['kepid'] == int(kepid)].values
    time_errors = labels_file['tce_time0_err'].loc[labels_file['kepid'] == int(kepid)].values
    loc_rowids = labels_file['loc_rowid'].loc[labels_file['kepid'] == int(kepid)].values
    i = 0
    for time_of_tce in time_of_tces:
        print(time_of_tce)
        for tce_fits_file in tce_fits_files:
            lc_segment = fits.open(tce_fits_file)
            lc_segment_data = lc_segment[1].data
            bjdrefi = lc_segment[1].header['BJDREFI']
            bjdreff = lc_segment[1].header['BJDREFF']
            times = lc_segment_data['time']
            bjds = times + bjdrefi + bjdreff
            #bjds = bjds[np.logical_not(np.isnan(bjds))]
            bjd_min = bjds[0]
            bjd_max = bjds[-1]
            print("["+str(bjd_min)+","+str(bjd_max)+"]")
            if time_of_tce > bjd_min and time_of_tce < bjd_max or (time_of_tce == bjd_min or time_of_tce == bjd_max):
                pdcsap_fluxes = lc_segment_data['PDCSAP_FLUX']
                pdcsap_fluxes_90 = pd.DataFrame(pdcsap_fluxes).quantile(.8)
                #print(pd.DataFrame(pdcsap_fluxes).quantile(.1))
                print(pdcsap_fluxes_90)
                file_path = tce_fits_file.split('/')
                time_inf_lim = time_of_tce-time_errors[i]
                time_sup_lim = time_of_tce+time_errors[i]
                plt.plot(bjds,pdcsap_fluxes)
                plt.title("Curva de luz en archivo "+file_path[-1]+". tce_kepid="+str(kepid)+". loc_rowid="+str(loc_rowids[i]))
                plt.axvline(x = time_of_tce, color = 'b')
                plt.axvline(x = time_inf_lim, color = 'r')
                plt.axvline(x = time_sup_lim, color = 'g')
                plt.axhline(y=pdcsap_fluxes_90, color = 'g',linestyle='-')
                plt.show()
        i=i+1

def export_dataset():
    loc_rowids = []
    tce_kepids = []
    file_names = []
    pdcsap_kurtosis = []
    pdcsap_skewness = []
    pdcsap_autocorrelation = []
    pdcsap_entropy = []
    pdcsap_wavelet_length = []
    grad_kurtosis = []
    grad_skewness = []
    grad_autocorrelation = []
    grad_entropy = []
    grad_wavelet_length = []
    fft_kurtosis = []
    fft_skewness = []
    fft_autocorrelation = []
    fft_entropy = []
    fft_wavelet_length = []
    labels = []
    tce_groups = glob.glob(kepler_data_path+"/*")
    for tce_group in tce_groups:
        tces = glob.glob(tce_group+"/*")
        for tce in tces:
            tce_fits_files = glob.glob(tce+"/*")
            tce_kepid = tce.split('/')[-1]
            tce_labels = []
            original_labels = labels_file['av_training_set'].loc[labels_file['kepid'] == int(tce_kepid)].values
            for original_label in original_labels:
                if original_label == 'PC':
                    label = 1
                else:
                    label = 0
                tce_labels.append(label)
            time_of_tces = labels_file['tce_time0'].loc[labels_file['kepid'] == int(tce_kepid)].values
            rowids = labels_file['loc_rowid'].loc[labels_file['kepid'] == int(tce_kepid)].values
            i = 0
            #tce_light_curve = np.array([])
            #bjds_light_curves = np.array([])
            for time_of_tce in time_of_tces:
                for tce_fits_file in tce_fits_files:
                    error = False
                    try:
                        lc_segment = fits.open(tce_fits_file)
                        lc_segment_data = lc_segment[1].data
                    except:
                        error=True
                        print("Hubo un error leyendo el archivo "+tce_fits_file+", podria estar corrupto")
                    if(not error):
                        bjdrefi = lc_segment[1].header['BJDREFI']
                        bjdreff = lc_segment[1].header['BJDREFF']
                        times = lc_segment_data['time']
                        bjds = times + bjdrefi + bjdreff
                        bjd_series = pd.Series(bjds)
                        bjd_series = bjd_series.interpolate(method='from_derivatives')
                        bjds = bjd_series.to_numpy()
                        #bjds = bjds[np.logical_not(np.isnan(bjds))]
                        bjd_min = bjds[0]
                        bjd_max = bjds[-1]
                        if time_of_tce > bjd_min and time_of_tce < bjd_max or (time_of_tce == bjd_min or time_of_tce == bjd_max):
                            loc_rowids.append(rowids[i])
                            tce_kepids.append(tce_kepid)
                            file_name = tce_fits_file.split('/')[-1]
                            file_names.append(file_name)
                            pdcsap_fluxes = lc_segment_data['PDCSAP_FLUX']
                            pdcsap_series = pd.Series(pdcsap_fluxes)
                            pdcsap_series = pdcsap_series.interpolate(method='from_derivatives')
                            pdcsap_fluxes = pdcsap_series.to_numpy()
                            #pdcsap_fluxes = pdcsap_fluxes[np.logical_not(np.isnan(pdcsap_fluxes))]
                            pdcsap_fluxes = (pdcsap_fluxes - np.min(pdcsap_fluxes)) / (np.max(pdcsap_fluxes) - np.min(pdcsap_fluxes))
                            grad_fluxes = get_gradient(pdcsap_fluxes)
                            fft_fluxes = np.abs(scipy.fftpack.fft(pdcsap_fluxes))
                            #Feature extraction
                            #Normalized light curve
                            pdcsap_kurtosis.append(kurtosis(pdcsap_fluxes))
                            pdcsap_skewness.append(skew(pdcsap_fluxes))
                            pdcsap_autocorrelation.append(np.sum(np.correlate(pdcsap_fluxes, pdcsap_fluxes, mode="full")))
                            pdcsap_entropy.append(entropy((pdcsap_fluxes+1),base=2))
                            pdcsap_wavelet_length.append(get_wavelet_length(pdcsap_fluxes))
                            #Light curve gradient
                            grad_kurtosis.append(kurtosis(grad_fluxes))
                            grad_skewness.append(skew(grad_fluxes))
                            grad_autocorrelation.append(np.sum(np.correlate(grad_fluxes, grad_fluxes, mode="full")))
                            grad_entropy.append(entropy((np.array(grad_fluxes)+1),base=2))
                            grad_wavelet_length.append(get_wavelet_length(grad_fluxes))
                            #Light curve Fourier Transform
                            fft_kurtosis.append(kurtosis(fft_fluxes))
                            fft_skewness.append(skew(fft_fluxes))
                            fft_autocorrelation.append(np.sum(np.correlate(fft_fluxes, fft_fluxes, mode="full")))
                            fft_entropy.append(entropy((np.array(fft_fluxes)+1),base=2))
                            fft_wavelet_length.append(get_wavelet_length(fft_fluxes))
                            
                            labels.append(tce_labels[i])
                            break
                print("TCE "+tce_kepid+" processed")
                i=i+1

    dict_dataset = {"loc_rowid:": loc_rowids,
                "tce_kepid:": tce_kepids,
                "file_name:": file_names,
                
                "Kurtosis": pdcsap_kurtosis, 
                "Skewness": pdcsap_skewness, 
                "Autocorrelation": pdcsap_autocorrelation, 
                "Entropy": pdcsap_entropy,
                "WL":pdcsap_wavelet_length,

                "Kurtosis2": grad_kurtosis, 
                "Skewness2": grad_skewness, 
                "Autocorrelation2": grad_autocorrelation, 
                "Entropy2": grad_entropy, 
                "WL2":grad_wavelet_length,

                "Kurtosis3": fft_kurtosis, 
                "Skewness3": fft_skewness, 
                "Autocorrelation3": fft_autocorrelation, 
                "Entropy3": fft_entropy, 
                "WL3":fft_wavelet_length,

                "target":labels}

    df_dataset = pd.DataFrame(dict_dataset)
    df_dataset = df_dataset.sort_values(by='target',ascending=False)
    df_dataset = df_dataset.drop_duplicates(subset=["Kurtosis","Skewness","Autocorrelation","Entropy","WL","Kurtosis2","Skewness2","Autocorrelation2","Entropy2","WL2","Kurtosis3","Skewness3","Autocorrelation3","Entropy3","WL3"])
    df_dataset.to_csv(kepler_out_data_path+"/output_dataset.csv", sep=';')
