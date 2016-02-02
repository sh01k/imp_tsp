# Overview
A python CLI script for measuring an impulse response with time-stretched pulse (TSP) signal.
`pyaudio`, `numpy` and `matplotlib` are required.

## Usage
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

If you want to check audio device information, run
`python audio_io_view.py`

## References
[Y. Suzuki, F. Asano, H. Kim, and T. Sone, "An optimum computer‐generated pulse signal suitable for the measurement of very long impulse responses," The Journal of Acoustical Society of America, vol. 97, no. 2, pp. 1119-1123, 1995.](http://scitation.aip.org/content/asa/journal/jasa/97/2/10.1121/1.412224)

## Author
[Shoichi Koyama](http://www.sh01.org/)
