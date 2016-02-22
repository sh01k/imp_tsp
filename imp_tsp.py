# -*- coding: utf-8 -*-

import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import wave
import sys
import argparse
import tsp_gen

# Parameters
chunk = 512 #length of chunk for pyaudio
max_amp = 8191#32767 #maximum amplitude for output

# Argument parser
parser = argparse.ArgumentParser(description='Measure impulse response with time-stretched pulse signal.')
parser.add_argument('-l', '--length', type=int, nargs=1, required=True, help='TSP length')
parser.add_argument('-f', '--sampfreq', type=int, nargs=1, required=True, help='Sampling frequency')
parser.add_argument('-o', '--output', type=int, nargs=1, help='Output channel number')
parser.add_argument('-i', '--input', type=int, nargs=1, help='Number of input channels')
parser.add_argument('-p', '--plot', type=int, nargs=1, help='Plot figure flag')
parser.add_argument('-c', '--channel', type=int, nargs=1, help='Channel number for plot')
parser.add_argument('-s', '--sync', type=int, nargs=1, help='Number of synchronous additions')
parser.add_argument('-w', '--wave', type=int, nargs=1, help='Output wave file flag')
parser.add_argument('-a', '--audio', type=int, nargs=1, help='Index of audio device')
parser.add_argument('-d', '--direc', type=int, nargs=1, help='TSP up or down (0: up, 1: down)')
parser.add_argument('-e', '--eval', type=int, nargs=1, help='TSP evaluation flag')

args = parser.parse_args()

# TSP length
if np.log2(args.length) != int(np.log2(args.length)):
    print "TSP length must be power of 2"
    sys.exit()
else:
    tsp_len = args.length[0]
    fftlen = tsp_len

# Sampling frequency
if args.sampfreq[0]<=0:
    print "Invalied argument for sampfreq"
    sys.exit()
else:
    Fs = args.sampfreq[0]

# Channel number of TSP output
if args.output!=None and args.output[0]<=0:
    print "Invalid argument for output"
    sys.exit()
elif args.output!=None:
    out_channel = args.output[0]
else:
    out_channel = 1

# Number of channels for measurement
if args.input!=None and args.output[0]<=0:
    print "Invalid argument for input"
    sys.exit()
elif args.input!=None:
    n_in_channel = args.input[0]
else:
    n_in_channel = 1

# Plot figure flag
if args.plot!=None and args.plot[0] not in (0,1):
    print "Invalid argument for plot"
    sys.exit()
elif args.plot!=None:
    flg_fig = args.plot
else:
    flg_fig = 1

# Channel number for plot
if args.channel!=None and (args.channel[0]<=0 or args.channel[0]>n_in_channel):
    print "Invalid argument for channel"
    sys.exit()
elif args.channel!=None:
    plt_ch = args.channel[0]-1
else:
    plt_ch = 0

# Number of synchronous additions
if args.sync!=None and args.sync[0]<=0:
    print "Invalid argument for sync"
    sys.exit()
elif args.sync!=None:
    nsync = args.sync[0]
else:
    nsync = 4

# Output wavefile flag
if args.wave!=None and args.wave[0] not in (0,1):
    print "Invalid argument for wave"
    sys.exit()
elif args.plot!=None:
    flg_dump = args.wave
else:
    flg_dump = 1

# Index of audio device
if args.audio!=None and args.audio[0]<=0:
    print "Invalied argument for audio"
    sys.exit()
elif args.audio!=None:
    dev_id = args.audio[0]
else:
    dev_id = -1 #default audio device

# TSP up or down
if args.direc!=None and args.direc[0] not in (0,1):
    print "Invalid argument for direc"
    sys.exit()
elif args.direc!=None:
    flg_ud = args.direc[0]
else:
    flg_ud = 1

# TSP evaluation flag
if args.eval!=None and args.eval[0] not in (0,1):
    print "Invalid argument for eval"
    sys.exit()
elif args.eval!=None:
    flg_eval = args.eval[0]
else:
    flg_eval = 0

print "- TSP length: %d" % tsp_len
print "- Sampling frequency: %d" % Fs
print "- Output channel: %d" % out_channel
print "- Number of input channels: %d" % n_in_channel

nchannel = max(out_channel, n_in_channel) #number of input and output channels
t = [float(ind)/float(Fs) for ind in range(0,fftlen)] #time [s]

# Audio device information
pa = pyaudio.PyAudio() #initialize pyaudio

if dev_id<0: #default audio device
    in_dev_info = pa.get_default_input_device_info()
    out_dev_info = pa.get_default_output_device_info()
else:
    in_dev_info = pa.get_device_info_by_index(dev_id)
    out_dev_info = in_dev_info

print "- Device (Input): %s, SampleRate: %dHz, MaxInputChannels: %d" % (in_dev_info['name'],int(in_dev_info['defaultSampleRate']),int(in_dev_info['maxInputChannels']))
print "- Device (Output): %s, SampleRate: %dHz, MaxOutputChannels: %d" % (out_dev_info['name'],int(out_dev_info['defaultSampleRate']),int(out_dev_info['maxOutputChannels']))

# Check audio device support
if pa.is_format_supported(Fs, in_dev_info['index'], nchannel, pyaudio.paInt16, out_dev_info['index'], nchannel, pyaudio.paInt16) == False:
    print "Error: audio driver does not support current setting"
    sys.exit()

# Generate TSP signal
tsp_sig, itsp_sig = tsp_gen.get_tsp(tsp_len, Fs, flg_ud, flg_eval)

max_tsp_sig = np.max(np.absolute(tsp_sig))
tsp_data = np.array([int(s/float(max_tsp_sig)*float(max_amp)) for s in tsp_sig], dtype=np.int16)

if flg_dump:
    wf = wave.open('tsp_out.wav','wb')
    wf.setparams((1, 2, Fs, tsp_len, 'NONE', 'not compressed'))
    wf.writeframesraw(tsp_data.tostring())
    wf.close()

# Output data of TSP signal
tsp_data_sync = np.zeros((nchannel,tsp_len*(nsync+1)), dtype=np.int16)
for i in np.arange(nsync):
    tsp_data_sync[out_channel-1][i*tsp_len:(i+1)*tsp_len] = tsp_data

tsp_out = (tsp_data_sync.T).reshape((tsp_len*(nsync+1)*nchannel,1))

# Open stream
if dev_id<0:
    stream = pa.open(format=pyaudio.paInt16,
                    channels=nchannel,
                    rate=Fs,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
else:
    stream = pa.open(format=pyaudio.paInt16,
                    channels=nchannel,
                    rate=Fs,
                    input=True,
                    output=True,
                    input_device_index=dev_id,
                    output_device_index=dev_id,
                    frames_per_buffer=chunk)

nframe = int(np.ceil(tsp_len*(nsync+1)/chunk)) #number of frames

# Stream I/O
pa_indata = []
for i in np.arange(nframe):
    pa_input = stream.read(chunk)
    pa_indata.append(pa_input)
    pa_outdata = tsp_out[i*nchannel*chunk:(i+1)*nchannel*chunk]
    pa_output = stream.write(pa_outdata.tostring())

# Stop stream
stream.stop_stream()
stream.close()

# Close PyAudio
pa.terminate()

tsp_rcv_raw = ''.join(pa_indata) #concatenate frames

if flg_dump:
    wf = wave.open('tsp_in.wav','wb')
    wf.setparams((nchannel, 2, Fs, tsp_len*(nsync+1), 'NONE', 'not compressed'))
    wf.writeframesraw(b''.join(tsp_rcv_raw))
    wf.close()

# Split channels
tsp_rcv_data = np.frombuffer(tsp_rcv_raw, dtype=np.int16)
tsp_rcv_sig = np.zeros((n_in_channel,tsp_len*(nsync+1)),dtype=np.float64)
for i in np.arange(n_in_channel):
    amp_av = 0.0
    for j in np.arange(tsp_len*(nsync+1)):
        amp_av = amp_av + (tsp_rcv_data[j*nchannel+i].astype(np.float64))**2/(tsp_len*(nsync+1))
        tsp_rcv_sig[i][j] = tsp_rcv_data[j*nchannel+i].astype(np.float64)/float(max_amp)
    if amp_av<1.0:
        print "Amplitude of channel #%d is too small" % (i+1)

# Synchronous addition
tsp_rcv_sum = np.zeros((n_in_channel,tsp_len), dtype=np.float64)
for i in np.arange(n_in_channel):
    for j in np.arange(nsync+1):
        tsp_rcv_sum[i][:] = tsp_rcv_sum[i][:] + tsp_rcv_sig[i][j*tsp_len:(j+1)*tsp_len]/float(nsync)

# Calculate impulse response
imp = np.zeros((n_in_channel,fftlen),dtype=np.float64)
for i in np.arange(n_in_channel):
    imp_spec = np.fft.fft(tsp_rcv_sum[i][:],fftlen)*np.fft.fft(itsp_sig,fftlen)
    imp[i][:] = (np.fft.ifft(imp_spec,fftlen)).real

# Output impulse response
fname_imp = "imp_fs%d_ch%d.bin" % (int(Fs),int(n_in_channel))
fbin = open(fname_imp,'wb')
fbin.write(((imp.T).reshape((n_in_channel*fftlen,1))).tostring())
fbin.close()

# Draw figures
if flg_fig:
    stft_len = 256
    stft_overlap = 128
    stft_win = np.hamming(stft_len)

    plt.figure()
    pxx, stft_freq, stft_bins, stft_time = plt.specgram(tsp_rcv_sig[plt_ch][:], NFFT=stft_len, Fs=Fs, window=stft_win, noverlap=stft_overlap)
    plt.axis([0, tsp_len*(nsync+1)/Fs, 0, Fs/2])
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")

    plt.figure()
    plt.plot(tsp_rcv_sig[plt_ch][:])

    plt.figure()
    plt.plot(tsp_rcv_sum[plt_ch][:])

    plt.figure()
    plt.plot(imp[plt_ch][:])

    plt.show()
