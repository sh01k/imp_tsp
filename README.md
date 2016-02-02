imp_tsp.py
====
## Description
A python CLI script for measuring an impulse response with time-stretched pulse (TSP) signal.
[pyaudio](https://people.csail.mit.edu/hubert/pyaudio/), [numpy](http://www.numpy.org/), and [matplotlib](http://matplotlib.org/) are required.

## Usage
The main script is `imp_tsp.py`. The length of TSP signal (samples) and sampling frequency (Hz) must be specified.
* python imp_tsp.py
  * [-h]
  * -l TSP_LENGTH
  * -f SAMPLING_FREQUENCY
  * [-o OUTPUT_CHANNEL]
  * [-i NUMBER_OF_INPUT_CHANNELS]
  * [-p PLOT_FIGURE_FLAG]
  * [-c PLOT_CHANNEL_NUMBER]
  * [-s NUMBER_OF_SYNCRONOUS_ADDITION]
  * [-w OUTPUT_WAVEFILE_FLAG]
  * [-a AUDIO_DEVICE_INDEX]
  * [-d TSP_DIRECTION]
  * [-e TSP_EVALUATION]

For example, when the length of TSP signal is 65536 samples and the sampling frequency is 48kHz, run:  
`python imp_tsp.py -l 65536 -f 48000`

To specify the output channel and the number of input channels, add "-o" and "-i" arguments:  
`python imp_tsp.py -l 65536 -f 48000 -o 2 -i 4`

If you want to check audio device information, run  
`python audio_io_view.py`

## Requirements
- [pyaudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [numpy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)

## References
- [Y. Suzuki, F. Asano, H. Kim, and T. Sone, "An optimum computer‐generated pulse signal suitable for the measurement of very long impulse responses," The Journal of Acoustical Society of America, vol. 97, no. 2, pp. 1119-1123, 1995.](http://scitation.aip.org/content/asa/journal/jasa/97/2/10.1121/1.412224)

## Author
[Shoichi Koyama](http://www.sh01.org/)
