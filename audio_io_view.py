# -*- coding: utf-8 -*-

import pyaudio

def audio_dump():
    pa = pyaudio.PyAudio()
    num_api = pa.get_host_api_count()
    num_dev = pa.get_device_count()

    api_info = [[]]
    for i in range(0,num_api):
        api_info[i] = pa.get_host_api_info_by_index(i)

    for i in range(0,num_dev):
        dev_info = pa.get_device_info_by_index(i)
        print "Index: %d" % i
        print " -DeviceName: %s\n -HostName: %s\n -SampleRate: %dHz\n -MaxInputChannels: %d\n -MaxOutputChannels: %d" % (dev_info['name'], api_info[dev_info['hostApi']]['name'], int(dev_info['defaultSampleRate']), int(dev_info['maxInputChannels']), int(dev_info['maxOutputChannels']))

audio_dump()
