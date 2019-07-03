import sys
import time
import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt

class imptsp:
    def __init__(self,Fs,tsp_len,nchannel=1,dev_id=-1,dbg_ch=1,flg_fig=0,flg_dump=0,nsync=3,flg_ud=1,flg_eval=0):
        self.chunk = 512 #length of chunk for pyaudio
        self.max_amp = 8191 #32767 #maximum amplitude for output
        self.format = pyaudio.paInt16 #format

        self.Fs = Fs #sampling frequency
        self.tsp_len = tsp_len #TSP length
        self.nchannel = nchannel #number of channels
        self.dev_id = dev_id #index of audio device
        self.dbg_ch = dbg_ch-1 #channel number for debug
        self.flg_fig = flg_fig #plot figure flag
        self.flg_dump = flg_dump #output wavefile flag
        self.nsync = nsync #number of synchronous additions
        self.flg_ud = flg_ud #TSP up or down
        self.flg_eval = flg_eval #TSP evaluation flag

        self.fftlen = self.tsp_len # FFT length
        self.stft_len = 256 #STFT length
        self.stft_overlap = 128 #STFT overlap length
        self.stft_win = np.hamming(self.stft_len)

        # Sampling frequency
        if self.Fs<=0:
            print("Invalid sampling frequency")
            self.usage()
            return

        #TSP length
        if np.log2(self.tsp_len) != int(np.log2(self.tsp_len)):
            print("TSP length must be power of 2")
            self.usage()
            return
        elif self.tsp_len<512:
            print("TSP length is too small")
            self.usage()
            return

        # Number of channels
        if self.nchannel<=0:
            print("Invalid number of channels")
            self.usage()
            return

        # Index of audio device
        if self.dev_id<-1:
            print("Invalied index of audio device")
            self.usage()
            return

        # Channel number for debug
        if (self.dbg_ch<0 or self.dbg_ch>=self.nchannel):
            print("Invalid channel number for debug")
            self.usage()
            return

        # Plot figure flag
        if self.flg_fig not in (0,1):
            print("Invalid plot figure flag")
            self.usage()
            return

        # Output wavefile flag
        if self.flg_dump not in (0,1):
            print("Invalid output wavefile flag")
            self.usage()
            return

        # Number of synchronous additions
        if self.nsync<=0:
            print("Invalid number of synchronous additions")
            self.usage()
            return

        # TSP up or down
        if self.flg_ud not in (0,1):
            print("Invalid TSP up or down")
            self.usage()
            return

        # TSP evaluation flag
        if self.flg_eval not in (0,1):
            print("Invalid TSP evaluation flag")
            self.usage()
            return

        # Format
        if self.format == pyaudio.paInt16:
            self.format_np = np.int16
        elif self.format == pyaudio.paInt32:
            self.format_np = np.int32
        elif self.format == pyaudio.paInt8:
            self.format_np = np.int8
        elif self.format == pyaudio.paUInt8:
            self.format_np = np.uint8
        elif self.format == pyaudio.paFloat32:
            self.format_np = np.float32
        else:
            print("Invalid format")
            self.usage()
            return

        print("- TSP length: %d" % self.tsp_len)
        print("- Sampling frequency: %d" % self.Fs)
        print("- Number of channels: %d" % self.nchannel)
        print("- Number of synchronous additions: %d" % self.nsync)

        self.t = [float(ind)/float(self.Fs) for ind in (0,self.fftlen)] #time [s]

        # Audio device information
        self.pa = pyaudio.PyAudio() #initialize pyaudio
        if self.dev_id>=0:
            in_dev_info = self.pa.get_device_info_by_index(self.dev_id)
            out_dev_info = in_dev_info
        else: #default audio device
            in_dev_info = self.pa.get_default_input_device_info()
            out_dev_info = self.pa.get_default_output_device_info()

        print("- Device (Input): %s, SampleRate: %dHz, MaxInputChannels: %d" % (in_dev_info['name'],int(in_dev_info['defaultSampleRate']),int(in_dev_info['maxInputChannels'])))
        print("- Device (Output): %s, SampleRate: %dHz, MaxOutputChannels: %d" % (out_dev_info['name'],int(out_dev_info['defaultSampleRate']),int(out_dev_info['maxOutputChannels'])))

        # Check audio device support
        if self.pa.is_format_supported(self.Fs, in_dev_info['index'], self.nchannel, self.format, out_dev_info['index'], self.nchannel, self.format) == False:
            print("Error: audio driver does not support current setting")
            return None

        # Generate TSP signal
        self.tsp_sig, self.itsp_sig = self.get_tsp()

        max_tsp_sig = np.max(np.absolute(self.tsp_sig))
        self.tsp_data = np.array([int(s/float(max_tsp_sig)*float(self.max_amp)) for s in self.tsp_sig], dtype=self.format_np)

        if self.flg_dump:
            wf = wave.open('tsp_out.wav','wb')
            wf.setparams((1, self.pa.get_sample_size(self.format), self.Fs, self.tsp_len, 'NONE', 'not compressed'))
            wf.writeframesraw(self.tsp_data.tostring())
            wf.close()

        self.ifrm = 0
        self.pa_indata = []
        self.tsp_out = np.zeros((self.tsp_len*(self.nsync+1)*self.nchannel,1), dtype=self.format_np)

        # Open stream
        if self.dev_id<0:
            self.stream = self.pa.open(format=self.format,
                                       channels=self.nchannel,
                                       rate=self.Fs,
                                       input=True,
                                       output=True,
                                       frames_per_buffer=self.chunk,
                                       stream_callback=self.callback)
        else:
            self.stream = self.pa.open(format=self.format,
                                       channels=self.nchannel,
                                       rate=self.Fs,
                                       input=True,
                                       output=True,
                                       input_device_index=self.dev_id,
                                       output_device_index=self.dev_id,
                                       frames_per_buffer=self.chunk,
                                       stream_callback=self.callback)

    def usage(self):
        print("[Usage]")
        print("  > imp = imptsp(Fs,tsp_len,nchannel,dev_id,dbg_ch,flg_fig,flg_dump,nsync,flg_ud,flg_eval)")
        print("  > (ir,tsp) = imp.get_imp(in_channel,out_channel)")
        print("  > imp.terminate()")
        print("  - Fs: sampling frequency\n  - tsp_len: TSP length\n  - nchannel: number of channels\n  - dev_id: index of audio device\n  - dbg_ch: channel number for debug\n  - flg_fig:plot figure flag\n  - flg_dump: output wavefile flag\n  - nsync: number of synchronous additions\n  - flg_ud: TSP up or down (0 or 1)\n  - flg_eval: TSP evaluation flag\n  - in_channel: list of input channels\n  - out_channel: channel number for output")
        return 0

    def get_imp(self,in_channel,out_channel):
        n_in_channel = len(in_channel)

        for i in range(n_in_channel):
            if in_channel[i]-1 not in np.arange(self.nchannel):
                print("Invalid channel number of input")
                self.usage()
                return 1
            else:
                in_channel[i] = in_channel[i]-1

        out_channel = out_channel-1 #channel number of output

        if out_channel >= self.nchannel | out_channel < 0:
            print("Invalid channel number of output")
            self.usage()
            return 1

        if self.flg_fig & self.dbg_ch not in in_channel:
            self.dbg_ch = in_channel[0]

        self.pa_indata = []

        # Output data of TSP signal
        tsp_data_sync = np.zeros((self.nchannel,self.tsp_len*(self.nsync+1)), dtype=self.format_np)
        for i in range(self.nsync):
            tsp_data_sync[out_channel,i*self.tsp_len:(i+1)*self.tsp_len] = self.tsp_data

        self.tsp_out = (tsp_data_sync.T).reshape((self.tsp_len*(self.nsync+1)*self.nchannel,1))

        nframe = int(np.ceil(self.tsp_len*(self.nsync+1)/self.chunk)) #number of frames

        self.ifrm = 0
        self.stream.start_stream()
        while self.ifrm<nframe:
            pass
        self.stream.stop_stream()

        tsp_rcv_raw = b''.join(self.pa_indata) #concatenate frames

        if self.flg_dump:
            wf = wave.open('tsp_in.wav','wb')
            wf.setparams((self.nchannel, self.pa.get_sample_size(self.format), self.Fs, self.tsp_len*(self.nsync+1), 'NONE', 'not compressed'))
            wf.writeframesraw(tsp_rcv_raw)
            wf.close()

        # Split channels
        tsp_rcv_data = np.frombuffer(tsp_rcv_raw, dtype=self.format_np)
        tsp_rcv_sig = np.zeros((n_in_channel,self.tsp_len*(self.nsync+1)),dtype=np.float64)
        for i in range(n_in_channel):
            amp_av = 0.0
            for j in range(self.tsp_len*(self.nsync+1)):
                amp_av = amp_av + (tsp_rcv_data[j*self.nchannel+in_channel[i]].astype(np.float64))**2/(self.tsp_len*(self.nsync+1))
                tsp_rcv_sig[i,j] = tsp_rcv_data[j*self.nchannel+in_channel[i]].astype(np.float64)/float(self.max_amp)
            if amp_av<1.0:
                print("Amplitude of channel #%d is too small" % (in_channel[i]+1))

        # Synchronous addition
        tsp_rcv_sum = np.zeros((n_in_channel,self.tsp_len), dtype=np.float64)
        for i in range(n_in_channel):
            for j in range(self.nsync+1):
                tsp_rcv_sum[i,:] = tsp_rcv_sum[i,:] + tsp_rcv_sig[i,j*self.tsp_len:(j+1)*self.tsp_len]/float(self.nsync)

        # Calculate impulse response
        imp = np.zeros((n_in_channel,self.fftlen),dtype=np.float64)
        for i in range(n_in_channel):
            imp_spec = np.fft.fft(tsp_rcv_sum[i,:],self.fftlen)*np.fft.fft(self.itsp_sig,self.fftlen)
            imp[i,:] = (np.fft.ifft(imp_spec,self.fftlen)).real

        # Draw figures
        if self.flg_fig:
            #plt.figure()
            #plt.plot(tsp_data_sync[out_channel,:])

            plt.figure()
            pxx, stft_freq, stft_bins, stft_time = plt.specgram(tsp_rcv_sig[in_channel[self.dbg_ch],:], NFFT=self.stft_len, Fs=self.Fs, window=self.stft_win, noverlap=self.stft_overlap)
            plt.axis([0, self.tsp_len*(self.nsync+1)/self.Fs, 0, self.Fs/2])
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")

            plt.figure()
            plt.plot(tsp_rcv_sig[in_channel[self.dbg_ch],:])

            #plt.figure()
            #plt.plot(tsp_rcv_sum[in_channel[self.dbg_ch],:])

            plt.figure()
            plt.plot(imp[in_channel[self.dbg_ch],:])

            plt.show()

        return (imp, tsp_rcv_sig[in_channel[self.dbg_ch],:])

    def terminate(self):
        self.stream.close()
        self.pa.terminate()

    def callback(self, in_data, frame_count, time_info, status):
        self.pa_indata.append(in_data)
        pa_outdata = self.tsp_out[self.ifrm*self.nchannel*self.chunk:(self.ifrm+1)*self.nchannel*self.chunk]
        self.ifrm = self.ifrm+1
        return (pa_outdata.tostring(), pyaudio.paContinue)

    def get_tsp(self):
        # TSP parameters
        N_set = [512, 1024, 2048, 4096, 8192, 16384]
        stretch_set = [7, 10, 12, 13, 14, 15]

        if self.tsp_len in N_set:
            stretch = float(stretch_set[N_set.index(self.tsp_len)])
        elif self.tsp_len>16384:
            stretch = 15.0

        M = int((stretch/32.0)*float(self.tsp_len))
        t = [float(ind)/float(self.Fs) for ind in range(0,self.tsp_len)]

        tsp_spec = np.zeros(self.tsp_len, dtype=complex)
        itsp_spec = np.zeros(self.tsp_len, dtype=complex)

        tsp_spec[0] = 1
        tsp_spec[int(self.tsp_len/2)] = np.exp(float(self.flg_ud*2-1)*1j*float(M)*np.pi)
        itsp_spec[0] = 1.0/tsp_spec[0]
        itsp_spec[int(self.tsp_len/2)] = 1.0/tsp_spec[int(self.tsp_len/2)]

        for i in range(1,int(self.tsp_len/2)):
            tsp_spec[i] = np.exp(float(self.flg_ud*2-1)*1j*4*float(M)*np.pi*(float(i-1)**2)/(float(self.tsp_len)**2))
            itsp_spec[i] = 1.0/tsp_spec[i]
            tsp_spec[self.tsp_len-i] = np.conjugate(tsp_spec[i])
            itsp_spec[self.tsp_len-i] = 1.0/tsp_spec[self.tsp_len-i]

        tsp_sig = (np.fft.ifft(tsp_spec,self.tsp_len)).real
        itsp_sig = (np.fft.ifft(itsp_spec,self.tsp_len)).real

        # Circular shift
        if self.flg_ud == 1:
            tsp_sig = np.roll(tsp_sig, int(-(self.tsp_len/2-M)))
            itsp_sig = np.roll(itsp_sig, int(self.tsp_len/2-M))
        elif self.flg_ud == 0:
            tsp_sig = np.roll(tsp_sig, int(self.tsp_len/2-M))
            itsp_sig = np.roll(itsp_sig, int(-(self.tsp_len/2-M)))

        # Evaluation
        if self.flg_eval:
            print("Evaluating TSP signal...")

            imp_eval_spec = np.fft.fft(tsp_sig,self.tsp_len)*np.fft.fft(itsp_sig,self.tsp_len)
            imp_eval = np.fft.ifft(imp_eval_spec,self.tsp_len)
            imp_eval_power = 20*np.log10(np.roll(np.abs(imp_eval), int(self.tsp_len/2)))

            plt.figure()
            plt.plot(t, tsp_sig)
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")

            plt.figure()
            plt.plot(t, itsp_sig)
            plt.xlabel("Time [s]")
            plt.ylabel("Amplitude")

            plt.figure()
            pxx, stft_freq, stft_bin, stft_t = plt.specgram(tsp_sig, NFFT=self.stft_len, Fs=self.Fs, window=self.stft_win, noverlap=self.stft_overlap)
            plt.axis([0, self.tsp_len/self.Fs, 0, self.Fs/2])
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")

            plt.figure()
            plt.plot(imp_eval_power)
            plt.ylabel("[dB]")

            plt.show()

        return (tsp_sig, itsp_sig)

if __name__== '__main__':
    # Parameters
    in_channel = [1,2]
    out_channel = 1

    # Initialize
    imp = imptsp(Fs=48000,tsp_len=65536)

    # Measuring impulse response
    (ir,tsp) = imp.get_imp(in_channel,out_channel)

    # Terminate
    imp.terminate()
