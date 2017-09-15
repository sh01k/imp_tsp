imptsp.py
====
## Description
A python class for measuring an impulse response with time-stretched pulse (TSP) signal.
[pyaudio](https://people.csail.mit.edu/hubert/pyaudio/), [numpy](http://www.numpy.org/), and [matplotlib](http://matplotlib.org/) are required.

In `example.py`, the output file is written in MATLAB/Octave format using [scipy.io](https://docs.scipy.org/doc/scipy/reference/io.html). `imp_fsXXXX_chXX.mat`

## Usage
The main class file is `imptsp.py`. The sampling frequency (Hz) and length of TSP signal (samples) must be specified. Please refer to `example.py` for the usage.

First, the class file is imported in your python script:
```
from imptsp import imptsp
```

Initialize for the sampling frequency of 4.8kHz and TSP length of 65536 samples:
```
imp = imptsp(48000, 65536)
```

Measure the impulse response for input channel \#1 and \#2 and output channel \#1:
```
(ir,tsp) = imp.get_imp([1,2],1)
```
The impulse response data is in `ir` and the recorded TSP signal is in `tsp` for debugging.

Terminate:
```
imp.terminate()
```

If you want to check audio device information, run
```
python check_audio_dev.py
```

The other configuration for initialization is as follows:
* imptsp(Fs, tsp_len, [nchannel], [dev_id], [dbg_ch], [flg_fig], [flg_dump], [nsync], [flg_ud], [flg_eval])
  * Fs: sampling frequency
  * tsp_len: TSP length
  * nchannel: number of channels
  * dev_id: index of audio device
  * dbg_ch: channel number for debug
  * flg_fig:plot figure flag
  * flg_dump: output wavefile flag
  * nsync: number of synchronous additions
  * flg_ud: TSP up or down (0 or 1)
  * flg_eval: TSP evaluation flag
  * in_channel: list of input channels
  * out_channel: channel number for output

## Requirements
- [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [scipy](https://www.scipy.org/) (optional)

## References
- [Y. Suzuki, et al. "An optimum computer‐generated pulse signal suitable for the measurement of very long impulse responses," .J. Acoust. Soc. Am., vol. 97, no. 2, pp. 1119-1123, 1995.](http://scitation.aip.org/content/asa/journal/jasa/97/2/10.1121/1.412224)

## License
[MIT](https://github.com/sh01k/imp_tsp/blob/master/LICENSE)

## Author
[Shoichi Koyama](http://www.sh01.org/)
