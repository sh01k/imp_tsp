clear all;
close all;

dir = '.';

c=340.29;

Fs = 48000;
nchannel = 1;

fname = sprintf('%s/imp_fs%d_ch%d.bin',dir,Fs,nchannel);
fprintf('%s\n',fname);

fin = fopen(fname,'r'); 
[imp_data,len] = fread(fin,'double');
fclose(fin);

imp = reshape(imp_data,[nchannel,len/nchannel]);

figure;
plot(imp(1,:));