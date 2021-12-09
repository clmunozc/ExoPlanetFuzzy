from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import scipy
from scipy.stats import kurtosis, skew, entropy, norm
from scipy import fftpack

kepler_data_path = "/home/newpc1/Documentos/ExoplanetML/data/kepler"
kepler_out_data_path = "/home/newpc1/Documentos/ExoplanetML/data/kepler-processed-dataset"
labels_path = "/home/newpc1/Documentos/ExoplanetML/data/q1_q17_dr24_tce_labeled.csv"
labels_file = pd.read_csv(labels_path,skiprows=17)

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
                file_path = tce_fits_file.split('/')
                time_inf_lim = time_of_tce-time_errors[i]
                time_sup_lim = time_of_tce+time_errors[i]
                plt.plot(bjds,pdcsap_fluxes)
                plt.title("Curva de luz en archivo "+file_path[-1]+". tce_kepid="+str(kepid)+". loc_rowid="+str(loc_rowids[i]))
                plt.axvline(x = time_of_tce, color = 'b')
                plt.axvline(x = time_inf_lim, color = 'r')
                plt.axvline(x = time_sup_lim, color = 'g')
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
