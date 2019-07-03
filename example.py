import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from imptsp import imptsp

# Parameters
Fs = 48000
tsp_len = 65536

in_channel = [1,2]
out_channel = 1
n_in_channel = len(in_channel)

# Initialize
imp = imptsp(Fs,tsp_len)

# Measuring impulse response
(ir,tsp) = imp.get_imp(in_channel,out_channel)

# Plot
plt.figure()
plt.plot(tsp)

plt.figure()
pxx, stft_freq, stft_bins, stft_time = plt.specgram(tsp, NFFT=imp.stft_len, Fs=Fs, window=imp.stft_win, noverlap=imp.stft_overlap)
plt.axis([0, tsp_len*(imp.nsync+1)/Fs, 0, Fs/2])
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")

plt.figure()
plt.plot(ir[0,:])

plt.show()

# Save data
fname_imp = "imp_fs%d_ch%d.mat" % (int(Fs),int(out_channel))
sio.savemat(fname_imp,{'ir':ir})

# Terminate
imp.terminate()
