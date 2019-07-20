# Simple preprocess for original data
# Create statistical features and time series features with acoustic_data
# If you want to know the analysis for earthquake data, please read "LANL-Earthquake-Prediction.ipynb"

import os
import warnings
import multiprocessing as mp
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import scipy.signal as sg
from scipy.signal import hilbert, convolve, hann
from scipy import stats
from tsfresh.feature_extraction import feature_calculators
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# some global variables
NUM_PROCESSES = 6  # using 6 processes to create features
INPUT_DIR = 'input'  # original data dir
NUM_SEG_PER_PROCESS = 4000  # each process generate 4000 samples
NYQUIST_FREQ = 75000  # half of the highest frequence
MAX_FREQ = 20000  # frequence above 20000 is almost noise accroding to the EDA
FREQ_BAND = 2500  # bandpass filter width
SIGNAL_LEN = 150000  # the length of signal segment(the length of each test signal is 150000)
OUTPUT_DIR = 'output'  # save files output by this code


def split_raw_data(train_data):
    """
    split raw data to 6 slices, so that we can use multiprocess to create features
    :param train_data: original train data
    :return: None, outputs csv files
    """
    print("split_raw_data")
    for i in range(NUM_PROCESSES):
        split_data_len = int(len(train_data.index) / NUM_PROCESSES)
        df = train_data.iloc[split_data_len * i: split_data_len * (i + 1)]
        df.to_csv(os.path.join(INPUT_DIR, 'train_%d.csv' % i), index=False)
        del df


def generate_start_indices(train_data):
    """
    generate start indices for segment in train data after spliting
    :param train_data: original train data
    :return: None, outputs a csv file
    """
    print("generate_start_indices")
    max_start_index = int(len(train_data.index) / NUM_PROCESSES) - SIGNAL_LEN
    rnd_idx_mat = np.zeros((NUM_SEG_PER_PROCESS, NUM_PROCESSES))
    for i in range(NUM_PROCESSES):
        np.random.seed(i ** 2)
        start_indices = np.random.randint(0, max_start_index, size=NUM_SEG_PER_PROCESS, dtype=np.int32)
        rnd_idx_mat[:, i] = start_indices
    np.savetxt(os.path.join(INPUT_DIR, 'start_indices_matrix.csv'), rnd_idx_mat, fmt='%d', delimiter=',')


def des_bw_filter_lp(cutoff):
    """
    4 pole Butterworth IIR low pass filter
    :param cutoff: low pass frequence cutline
    :return: b, a: coefficients of filter
    """
    b, a = sg.butter(4, Wn=cutoff / NYQUIST_FREQ, btype='lowpass')
    return b, a


def des_bw_filter_hp(cutoff):
    """
    4 pole Butterworth IIR high pass filter
    :param cutoff: high pass frequence cutline
    :return: b, a: coefficients of filter
    """
    b, a = sg.butter(4, Wn=cutoff / NYQUIST_FREQ, btype='highpass')
    return b, a


def des_bw_filter_bp(low, high):
    """
    4 pole Butterworth IIR band pass filter
    :param cutoff: band pass frequence cutline
    :return: b, a: coefficients of filter
    """
    b, a = sg.butter(4, Wn=[low / NYQUIST_FREQ, high / NYQUIST_FREQ], btype='bandpass')
    return b, a


def calc_mean_change_rate(arr):
    """
    claculate mean change rate for a signal segment
    :param arr: signal segment
    :return: mean change rate
    """
    change_rate = (np.diff(arr) / arr[:-1]).values
    change_rate = change_rate[np.nonzero(change_rate)[0]]
    change_rate = change_rate[~np.isnan(change_rate)]
    change_rate = change_rate[change_rate != np.inf]
    change_rate = change_rate[change_rate != -np.inf]
    return np.mean(change_rate)


def add_trend_feature(arr, abs_values=False):
    """
    add a trend feature based on input array
    :param arr: a signal segment
    :param abs_values: only positive value or not
    :return: slope of the trend
    """
    index = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(index.reshape(-1, 1), arr)
    return lr.coef_[0]


def classic_sta_lta(x, length_sta, length_lta):
    """
    calculate short term average divided by long term average
    :param x: a signal segment
    :param length_sta: the length of short time average
    :param length_lta: the length of long time average
    :return: short term average divided by long term average
    """
    sta = np.cumsum(x ** 2)
    sta = np.asarray(sta, dtype=np.float)
    lta = sta.copy()
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta
    sta[:length_lta - 1] = 0
    dtiny = np.finfo(float).tiny
    lta[lta < dtiny] = dtiny
    return sta / lta


def create_features(seg_id, seg, X, st, end):
    """
    create features including fft features, statistical features and time series features
    :param seg_id: the ID for a sample
    :param seg: s signal segment
    :param X: train set features before creating these features
    :param st: the start index of the signal segment
    :param end: the end index of the signal segment
    :return: train set features after creating these features
    """
    try:
        # test set won't create these features because its seg_id is string
        X.loc[seg_id, 'seg_id'] = np.int32(seg_id)
        X.loc[seg_id, 'seg_start'] = np.int32(st)
        X.loc[seg_id, 'seg_end'] = np.int32(end)
    except ValueError:
        pass

    xc = pd.Series(seg['acoustic_data'].values)
    xcdm = xc - np.mean(xc)

    b, a = des_bw_filter_lp(cutoff=18000)
    xcz = sg.lfilter(b, a, xcdm)

    zc = np.fft.fft(xcz)
    zc = zc[:MAX_FREQ]

    # FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)

    freq_bands = [x for x in range(0, MAX_FREQ, FREQ_BAND)]
    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)
    phzFFT = np.arctan(imagFFT / realFFT)
    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0
    phzFFT[phzFFT == np.inf] = np.pi / 2.0
    phzFFT = np.nan_to_num(phzFFT)

    for freq in freq_bands:
        X.loc[seg_id, 'FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_BAND], 0.01)
        X.loc[seg_id, 'FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_BAND], 0.1)
        X.loc[seg_id, 'FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_BAND], 0.9)
        X.loc[seg_id, 'FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_BAND], 0.99)
        X.loc[seg_id, 'FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_BAND])
        X.loc[seg_id, 'FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_BAND])
        X.loc[seg_id, 'FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_BAND])

        X.loc[seg_id, 'FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_BAND])
        X.loc[seg_id, 'FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_BAND])

    X.loc[seg_id, 'FFT_Rmean'] = realFFT.mean()
    X.loc[seg_id, 'FFT_Rstd'] = realFFT.std()
    X.loc[seg_id, 'FFT_Rmax'] = realFFT.max()
    X.loc[seg_id, 'FFT_Rmin'] = realFFT.min()
    X.loc[seg_id, 'FFT_Imean'] = imagFFT.mean()
    X.loc[seg_id, 'FFT_Istd'] = imagFFT.std()
    X.loc[seg_id, 'FFT_Imax'] = imagFFT.max()
    X.loc[seg_id, 'FFT_Imin'] = imagFFT.min()

    X.loc[seg_id, 'FFT_Rmean_first_6000'] = realFFT[:6000].mean()
    X.loc[seg_id, 'FFT_Rstd__first_6000'] = realFFT[:6000].std()
    X.loc[seg_id, 'FFT_Rmax_first_6000'] = realFFT[:6000].max()
    X.loc[seg_id, 'FFT_Rmin_first_6000'] = realFFT[:6000].min()
    X.loc[seg_id, 'FFT_Rmean_first_18000'] = realFFT[:18000].mean()
    X.loc[seg_id, 'FFT_Rstd_first_18000'] = realFFT[:18000].std()
    X.loc[seg_id, 'FFT_Rmax_first_18000'] = realFFT[:18000].max()
    X.loc[seg_id, 'FFT_Rmin_first_18000'] = realFFT[:18000].min()

    del xcz
    del zc

    b, a = des_bw_filter_lp(cutoff=2500)
    xc0 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=2500, high=5000)
    xc1 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=5000, high=7500)
    xc2 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=7500, high=10000)
    xc3 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=10000, high=12500)
    xc4 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=12500, high=15000)
    xc5 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=15000, high=17500)
    xc6 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_bp(low=17500, high=20000)
    xc7 = sg.lfilter(b, a, xcdm)

    b, a = des_bw_filter_hp(cutoff=20000)
    xc8 = sg.lfilter(b, a, xcdm)

    sigs = [xc, pd.Series(xc0), pd.Series(xc1), pd.Series(xc2), pd.Series(xc3),
            pd.Series(xc4), pd.Series(xc5), pd.Series(xc6), pd.Series(xc7), pd.Series(xc8)]

    for i, sig in enumerate(sigs):
        X.loc[seg_id, 'mean_%d' % i] = sig.mean()
        X.loc[seg_id, 'std_%d' % i] = sig.std()
        X.loc[seg_id, 'max_%d' % i] = sig.max()
        X.loc[seg_id, 'min_%d' % i] = sig.min()

        X.loc[seg_id, 'mean_change_abs_%d' % i] = np.mean(np.diff(sig))
        X.loc[seg_id, 'mean_change_rate_%d' % i] = calc_mean_change_rate(sig)
        X.loc[seg_id, 'abs_max_%d' % i] = np.abs(sig).max()

        X.loc[seg_id, 'std_first_50000_%d' % i] = sig[:50000].std()
        X.loc[seg_id, 'std_last_50000_%d' % i] = sig[-50000:].std()
        X.loc[seg_id, 'std_first_10000_%d' % i] = sig[:10000].std()
        X.loc[seg_id, 'std_last_10000_%d' % i] = sig[-10000:].std()

        X.loc[seg_id, 'avg_first_50000_%d' % i] = sig[:50000].mean()
        X.loc[seg_id, 'avg_last_50000_%d' % i] = sig[-50000:].mean()
        X.loc[seg_id, 'avg_first_10000_%d' % i] = sig[:10000].mean()
        X.loc[seg_id, 'avg_last_10000_%d' % i] = sig[-10000:].mean()

        X.loc[seg_id, 'min_first_50000_%d' % i] = sig[:50000].min()
        X.loc[seg_id, 'min_last_50000_%d' % i] = sig[-50000:].min()
        X.loc[seg_id, 'min_first_10000_%d' % i] = sig[:10000].min()
        X.loc[seg_id, 'min_last_10000_%d' % i] = sig[-10000:].min()

        X.loc[seg_id, 'max_first_50000_%d' % i] = sig[:50000].max()
        X.loc[seg_id, 'max_last_50000_%d' % i] = sig[-50000:].max()
        X.loc[seg_id, 'max_first_10000_%d' % i] = sig[:10000].max()
        X.loc[seg_id, 'max_last_10000_%d' % i] = sig[-10000:].max()

        X.loc[seg_id, 'max_to_min_%d' % i] = sig.max() / np.abs(sig.min())
        X.loc[seg_id, 'max_to_min_diff_%d' % i] = sig.max() - np.abs(sig.min())
        X.loc[seg_id, 'count_big_%d' % i] = len(sig[np.abs(sig) > 500])

        X.loc[seg_id, 'mean_change_rate_first_50000_%d' % i] = calc_mean_change_rate(sig[:50000])
        X.loc[seg_id, 'mean_change_rate_last_50000_%d' % i] = calc_mean_change_rate(sig[-50000:])
        X.loc[seg_id, 'mean_change_rate_first_10000_%d' % i] = calc_mean_change_rate(sig[:10000])
        X.loc[seg_id, 'mean_change_rate_last_10000_%d' % i] = calc_mean_change_rate(sig[-10000:])

        X.loc[seg_id, 'q95_%d' % i] = np.quantile(sig, 0.95)
        X.loc[seg_id, 'q99_%d' % i] = np.quantile(sig, 0.99)
        X.loc[seg_id, 'q05_%d' % i] = np.quantile(sig, 0.05)
        X.loc[seg_id, 'q01_%d' % i] = np.quantile(sig, 0.01)

        X.loc[seg_id, 'abs_q95_%d' % i] = np.quantile(np.abs(sig), 0.95)
        X.loc[seg_id, 'abs_q99_%d' % i] = np.quantile(np.abs(sig), 0.99)
        X.loc[seg_id, 'abs_q05_%d' % i] = np.quantile(np.abs(sig), 0.05)
        X.loc[seg_id, 'abs_q01_%d' % i] = np.quantile(np.abs(sig), 0.01)

        X.loc[seg_id, 'trend_%d' % i] = add_trend_feature(sig)
        X.loc[seg_id, 'abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)
        X.loc[seg_id, 'abs_mean_%d' % i] = np.abs(sig).mean()
        X.loc[seg_id, 'abs_std_%d' % i] = np.abs(sig).std()

        X.loc[seg_id, 'mad_%d' % i] = sig.mad()
        X.loc[seg_id, 'kurt_%d' % i] = sig.kurtosis()
        X.loc[seg_id, 'skew_%d' % i] = sig.skew()
        X.loc[seg_id, 'med_%d' % i] = sig.median()

        X.loc[seg_id, 'Hilbert_mean_%d' % i] = np.abs(hilbert(sig)).mean()
        X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

        X.loc[seg_id, 'classic_sta_lta1_mean_%d' % i] = classic_sta_lta(sig, 500, 10000).mean()
        X.loc[seg_id, 'classic_sta_lta2_mean_%d' % i] = classic_sta_lta(sig, 5000, 100000).mean()
        X.loc[seg_id, 'classic_sta_lta3_mean_%d' % i] = classic_sta_lta(sig, 3333, 6666).mean()
        X.loc[seg_id, 'classic_sta_lta4_mean_%d' % i] = classic_sta_lta(sig, 10000, 25000).mean()

        X.loc[seg_id, 'Moving_average_700_mean_%d' % i] = sig.rolling(window=700).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_1500_mean_%d' % i] = sig.rolling(window=1500).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_3000_mean_%d' % i] = sig.rolling(window=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'Moving_average_6000_mean_%d' % i] = sig.rolling(window=6000).mean().mean(skipna=True)

        ewma = pd.Series.ewm
        X.loc[seg_id, 'exp_Moving_average_300_mean_%d' % i] = ewma(sig, span=300).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_3000_mean_%d' % i] = ewma(sig, span=3000).mean().mean(skipna=True)
        X.loc[seg_id, 'exp_Moving_average_30000_mean_%d' % i] = ewma(sig, span=30000).mean().mean(skipna=True)

        no_of_std = 3
        X.loc[seg_id, 'MA_700MA_std_mean_%d' % i] = sig.rolling(window=700).std().mean()
        X.loc[seg_id, 'MA_700MA_BB_high_mean_%d' % i] = (
                X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[
            seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_700MA_BB_low_mean_%d' % i] = (
                X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[
            seg_id, 'MA_700MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_std_mean_%d' % i] = sig.rolling(window=400).std().mean()
        X.loc[seg_id, 'MA_400MA_BB_high_mean_%d' % i] = (
                X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[
            seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_400MA_BB_low_mean_%d' % i] = (
                X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[
            seg_id, 'MA_400MA_std_mean_%d' % i]).mean()
        X.loc[seg_id, 'MA_1000MA_std_mean_%d' % i] = sig.rolling(window=1000).std().mean()

        X.loc[seg_id, 'iqr_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))
        X.loc[seg_id, 'q999_%d' % i] = np.quantile(sig, 0.999)
        X.loc[seg_id, 'q001_%d' % i] = np.quantile(sig, 0.001)
        X.loc[seg_id, 'ave10_%d' % i] = stats.trim_mean(sig, 0.1)

        X.loc[seg_id, 'num_peaks_10_%d' % i] = feature_calculators.number_peaks(sig, 10)
        X.loc[seg_id, 'cid_ce_1_%d' % i] = feature_calculators.cid_ce(sig, 1)  # time series complexity
        X.loc[seg_id, 'count_1000_0_%d' % i] = feature_calculators.range_count(sig, -1000, 0)
        X.loc[seg_id, 'binned_entropy_5_%d' % i] = feature_calculators.binned_entropy(sig, 5)
        X.loc[seg_id, 'binned_entropy_15_%d' % i] = feature_calculators.binned_entropy(sig, 15)

    # sliding window is a kind of filter, so this code is out of the cycle of band pass
    for windows in [10, 100, 1000]:
        x_roll_std = xc.rolling(windows).std().dropna()
        x_roll_mean = xc.rolling(windows).mean().dropna()

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = calc_mean_change_rate(x_roll_std)
        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = calc_mean_change_rate(x_roll_mean)
        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    return X

def build_train_set(pid):
    """
    build train set for each process
    :param pid: the process id
    :param train_data: original train data
    :return: None, output creatures and labels of trainset for each process
    """
    success = 1
    try:
        seg_start = int(pid * NUM_SEG_PER_PROCESS)
        start_indices = np.loadtxt(os.path.join(INPUT_DIR, 'start_indices_matrix.csv'),
                                   dtype=np.int32, delimiter=',')[:, pid]
        train_X = pd.DataFrame(dtype=np.float64)
        train_y = pd.DataFrame(dtype=np.float64, columns=['time_to_failure'])
        train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train_%d.csv' % pid),
                               dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
        for seg_id, start_index in zip(tqdm(range(seg_start, seg_start + NUM_SEG_PER_PROCESS)), start_indices):
            end_index = start_index + SIGNAL_LEN
            seg = train_df.iloc[start_index: end_index]
            train_X = create_features(seg_id, seg, train_X, start_index, end_index)
            train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
        train_X.to_csv(os.path.join(OUTPUT_DIR, 'train_X_%d.csv' % pid), index=False)
        train_y.to_csv(os.path.join(OUTPUT_DIR, 'train_y_%d.csv' % pid), index=False)

    except OSError:
        success = 0
    return success


def run_multiprocess():
    """
    the multiprocess code to speed up creating features
    :param train_data: original train data
    :return: None
    """
    pool = mp.Pool(processes=NUM_PROCESSES)
    results = [pool.apply_async(build_train_set, args=(pid, )) for pid in range(NUM_PROCESSES)]
    output = [result.get() for result in results]
    pool.close()
    pool.join()
    print(sum(output))


def merge_train_set():
    """
    merge the trainset of each process
    :return: None, output features and labels for train set
    """
    print("merge_train_set")
    df_X = pd.read_csv('train_X_%d.csv' % 0)
    df_y = pd.read_csv("train_y_%d.csv" % 0)
    for i in range(1, NUM_PROCESSES):
        temp_X = pd.read_csv("train_X_%d.csv" % i)
        df_X = pd.concat([df_X, temp_X], axis=0)
        temp_y = pd.read_csv("train_y_%d.csv" % i)
        df_y = pd.concat([df_y, temp_y], axis=0)

    df_X.to_csv(os.path.join(OUTPUT_DIR, 'train_X.csv'), index=False)
    df_y.to_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'), index=False)


def build_test_set():
    """
    create features for test set
    :return: None, output features for test set
    """
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_X.csv'))
    try:
        drop_cols = ['seg_id', 'seg_start', 'seg_end']
        train_X.drop(labels=drop_cols, axis=1, inplace=True)
    except ValueError:
        pass
    submission = pd.read_csv(os.path.join(INPUT_DIR, 'sample_submission.csv'), index_col='seg_id')
    test_X = pd.DataFrame(dtype=np.float64, columns=train_X.columns, index=submission.index)
    for seg_id in tqdm(test_X.index):
        seg = pd.read_csv('input/test/' + seg_id + '.csv')
        test_X = create_features(seg_id, seg, test_X, 0, 0)

    test_X.to_csv(os.path.join(OUTPUT_DIR, 'test_X.csv'), index=False)


def scale_data():
    """
    scale the features of train set and test set to fit the labels better
    :return: None, output the scaled features for train set and test set
    """
    print("scale_data")
    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_X.csv'))
    try:
        drop_cols = ['seg_id', 'seg_start', 'seg_end']
        train_X.drop(labels=drop_cols, axis=1, inplace=True)
    except ValueError:
        pass

    test_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_X.csv'))
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)
    test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)

    train_X.to_csv(os.path.join(OUTPUT_DIR, 'scaled_train_X.csv'), index=False)
    test_X.to_csv(os.path.join(OUTPUT_DIR, 'scaled_test_X.csv'), index=False)


if __name__ == "__main__":
    train_data = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'),
                             dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
    split_raw_data(train_data)
    generate_start_indices(train_data)
    run_multiprocess()
    merge_train_set()
    build_test_set()
    scale_data()
