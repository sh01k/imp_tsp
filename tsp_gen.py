# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def get_tsp(N, Fs, flg_ud=1, flg_eval=0):
    if np.log2(N) != int(np.log2(N)):
        print "TSP length must be power of 2"
        return 0
    elif N<512:
        print "TSP length is too small"
        return 0

    if flg_ud != 1 and flg_ud != 0:
        print "TSP up and down flag is invalied"
        return 0

    # TSP parameters
    N_set = [512, 1024, 2048, 4096, 8192, 16384]
    stretch_set = [7, 10, 12, 13, 14, 15]

    if N in N_set:
        stretch = float(stretch_set(N_set.index(N)))
    elif N>16384:
        stretch = 15.0

    M = int((stretch/32.0)*float(N))
    t = [float(ind)/float(Fs) for ind in range(0,N)]

    tsp_spec = np.zeros(N, dtype=complex)
    itsp_spec = np.zeros(N, dtype=complex)

    tsp_spec[0] = 1
    tsp_spec[N/2] = np.exp(float(flg_ud*2-1)*1j*float(M)*np.pi)
    itsp_spec[0] = 1.0/tsp_spec[0]
    itsp_spec[N/2] = 1.0/tsp_spec[N/2]

    for i in np.arange(1,N/2):
        tsp_spec[i] = np.exp(float(flg_ud*2-1)*1j*4*float(M)*np.pi*(float(i-1)**2)/(float(N)**2))
        itsp_spec[i] = 1.0/tsp_spec[i]
        tsp_spec[N-i] = np.conjugate(tsp_spec[i])
        itsp_spec[N-i] = 1.0/tsp_spec[N-i]

    tsp_sig = (np.fft.ifft(tsp_spec,N)).real
    itsp_sig = (np.fft.ifft(itsp_spec,N)).real

    # Circular shift
    if flg_ud == 1:
        tsp_sig = np.roll(tsp_sig, -(N/2-M))
        itsp_sig = np.roll(itsp_sig, N/2-M)
    elif flg_ud == 0:
        tsp_sig = np.roll(tsp_sig, N/2-M)
        itsp_sig = np.roll(itsp_sig, -(N/2-M))

    # Evaluation
    if flg_eval:
        print "Evaluating TSP signal..."

        imp_eval_spec = np.fft.fft(tsp_sig,N)*np.fft.fft(itsp_sig,N)
        imp_eval = np.fft.ifft(imp_eval_spec,N)
        imp_eval_power = 20*np.log10(np.roll(np.abs(imp_eval), N/2))

        plt.figure()
        plt.plot(t, tsp_sig)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        plt.figure()
        plt.plot(t, itsp_sig)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")

        stft_len = 256
        stft_overlap = 128
        stft_win = np.hamming(stft_len)

        plt.figure()
        pxx, stft_freq, stft_bin, stft_t = plt.specgram(tsp_sig, NFFT=stft_len, Fs=Fs, window=stft_win, noverlap=stft_overlap)
        plt.axis([0, N/Fs, 0, Fs/2])
        plt.xlabel("Time [s]")
        plt.ylabel("Frequency [Hz]")

        plt.figure()
        plt.plot(imp_eval_power)
        plt.ylabel("[dB]")

        #plt.show()

    return (tsp_sig, itsp_sig)
