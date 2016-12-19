import numpy as np
import os
from kss import CurveManager

def smooth_data(data, filt_len):
    mean_filter = np.asarray([1] * filt_len)
    mean_filter = mean_filter / np.sum(mean_filter)
    return np.convolve(data, mean_filter, mode='valid')

def windowed_std(data, window_size):
    std = []
    for i in range(int(len(data) / window_size)):
        std += [np.std(data[i*window_size:(i+1)*window_size])]
    return np.asarray(std)

def get_diff(std, window_size, filt_len):
    std_start = std[:int(150/window_size)]
    std_end = std[int(150/window_size):]

    filt_range = (int)(filt_len / 2) + 1
    filt = [1/coeff for coeff in range(1, filt_range)]
    filt = filt[::-1] + [0] + list(- np.asarray(filt))

    diff_start = np.convolve(std_start, filt, mode='valid')
    diff_end = np.convolve(std_end, filt, mode='valid')
    
    return diff_start, diff_end

def get_start_end(data, window_size, filt_len):

    data_smoothed = smooth_data(data, filt_len)
    
    data_std = windowed_std(data, window_size)
    
    diff_start, diff_end = get_diff(data_std, window_size, filt_len)
    
    start = int((np.argmin(diff_start) + filt_len - 1) * window_size) + filt_len - 1
    end = int((np.argmax(diff_end) + 150/window_size) * window_size) + filt_len - 1
    return start, end

def get_parameter_csv(base_path):
    cm_path = os.path.join(base_path, 'raw_cm.hdf5')
    dmvt_path = os.path.join(base_path, 'raw_dmvt.hdf5')
    
    cm = CurveManager.load(cm_path)
    key = 'DST_Temp_vor_STB'
    indexNoZeros = cm.index[cm.index['DST_Temp_vor_STB'] != 0]
    curves = cm.load_curves(key, indexNoZeros)
    
    with open('parameters.csv', 'w') as f:
    
        f.write('DMVT_ID, Mean, Variance\n')
        
        while(True):
            try:
                key, data = next(curves)
                pseudoNr=int(indexNoZeros[indexNoZeros.index == key]["DPRD_PSEUDO_NR"])
                start, end = get_start_end(data, 10, 3)
                cropped_data = data[start:end]
                data_mean = np.mean(cropped_data)
                data_var = np.var(cropped_data)
                f.write('{}, {}, {:.2f}, {:.2f}\n'.format(key, pseudoNr, data_mean, data_var))
                
            except Exception as e:
                f.close()
                break
