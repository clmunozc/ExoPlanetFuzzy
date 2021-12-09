from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
import scipy
from scipy.stats import kurtosis, skew, entropy, norm
from scipy import fftpack

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

kepler_data_path = "/home/newpc1/Documentos/ExoplanetML/data/kepler"
kepler_out_data_path = "/home/newpc1/Documentos/ExoplanetML/data/kepler-processed-dataset"
labels_path = "/home/newpc1/Documentos/ExoplanetML/data/q1_q17_dr24_tce_labeled.csv"
tce_groups = glob.glob(kepler_data_path+"/*")
labels_file = pd.read_csv(labels_path)

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

for tce_group in tce_groups:
    tces = glob.glob(tce_group+"/*")
    for tce in tces:
        tce_fits_files = glob.glob(tce+"/*")
        tce_kepid = tce.split('/')[-1]
        original_labels = labels_file['av_training_set'].loc[labels_file['kepid'] == int(tce_kepid)].values
        tce_labels = []
        for original_label in original_labels:
            if original_label == 'PC':
                label = 2
            else:
                label = 1
            tce_labels.append(label)
        #labels.append(label)
        time_of_tces = labels_file['tce_time0'].loc[labels_file['kepid'] == int(tce_kepid)].values
        #tce_light_curve = np.array([])
        #bjds_light_curves = np.array([])
        for i in range(len(time_of_tces)):
            time_of_tce = time_of_tces[i]
            for tce_fits_file in tce_fits_files:
                try:
                    print("Procesando archivo "+tce_fits_file+"...")
                    lc_segment = fits.open(tce_fits_file)
                    lc_segment_data = lc_segment[1].data
                    bjdrefi = lc_segment[1].header['BJDREFI']
                    bjdreff = lc_segment[1].header['BJDREFF']
                    times = lc_segment_data['time']
                    bjds = times + bjdrefi + bjdreff
                    bjds = bjds[np.logical_not(np.isnan(bjds))]
                    bjd_min = bjds[0]
                    bjd_max = bjds[-1]
                    if time_of_tce > bjd_min and time_of_tce < bjd_max or (time_of_tce == bjd_min or time_of_tce == bjd_max):
                        print("TCE encontrado en "+tce_fits_file+", extrayendo caracteristicas de curva de luz")
                        labels.append(tce_labels[i])
                        pdcsap_fluxes = lc_segment_data['PDCSAP_FLUX']
                        pdcsap_fluxes = pdcsap_fluxes[np.logical_not(np.isnan(pdcsap_fluxes))]
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
                        break
                except:
                    print("Hubo un error leyendo el archivo "+tce_fits_file+", podria estar corrupto")
                

dict_dataset = {"Kurtosis": pdcsap_kurtosis, 
              "Skewness": pdcsap_skewness, 
              "Autocorrelation": pdcsap_autocorrelation, 
              "S.Entropy": pdcsap_entropy,
              "WL":pdcsap_wavelet_length,

              "Kurtosis2": grad_kurtosis, 
              "Skewness2": grad_skewness, 
              "Autocorrelation2": grad_autocorrelation, 
              "S.Entropy2": grad_entropy, 
              "WL2":grad_wavelet_length,

              "Kurtosis3": fft_kurtosis, 
              "Skewness3": fft_skewness, 
              "Autocorrelation3": fft_autocorrelation, 
              "S.Entropy3": fft_entropy, 
              "WL3":fft_wavelet_length,

              "target":labels}

df_dataset = pd.DataFrame(dict_dataset)
df_dataset.to_csv(kepler_out_data_path+"/dataset.csv", sep='\t')

            #tce_light_curve = np.concatenate((tce_light_curve,pdcsap_fluxes),axis=0,dtype='float32')
            #bjds_light_curves = np.concatenate((bjds_light_curves,bjds),axis=0,dtype='float32')
        #tce_light_curve = tce_light_curve[np.logical_not(np.isnan(tce_light_curve))]
        #normalized_tce_light_curve = (tce_light_curve - np.min(tce_light_curve)) / (np.max(tce_light_curve) - np.min(tce_light_curve))
        

