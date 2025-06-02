import soundfile as sf
import numpy as np
import sys
import os
import re


def add_noise(noisedir, cleandir, snr):
    # noisy
    splitdir = re.split(r"\\", noisedir)
    wavdir = ""  # 所有wav文件所在路径
    for i in range(len(splitdir) - 1):
        wavdir += splitdir[i] + '/'
    noisydir = wavdir + "noisy/"  # 带噪语音存储路径
    os.mkdir(noisydir)
    # print()
    # noise
    for noisewav in os.listdir(noisedir):
        noise, fs = sf.read(noisedir + '/' + noisewav)
        noisy_splitdir = noisydir + "add_" + noisewav[:-4] + "/"
        os.mkdir(noisy_splitdir)
        print(noise)
        # clean
        for cleanwav in os.listdir(cleandir):
            clean, Fs = sf.read(cleandir + "/" + cleanwav)
            # add noise
            if fs == Fs and len(clean) <= len(noise):
                # 纯净语音能量
                cleanenergy = np.sum(np.power(clean, 2))
                # 随机索引与clean长度相同的noise信号
        ind = np.random.randint(1, len(noise) - len(clean) + 1)
        noiselen = noise[ind:len(clean) + ind]
        # 噪声语音能量
        noiseenergy = np.sum(np.power(noiselen, 2))
        # 噪声等级系数
        noiseratio = np.sqrt((cleanenergy / noiseenergy) / (np.power(10, snr * 0.1)))
        # 随机索引与clean长度相同的noise信号
        noisyAudio = clean + noise[ind:len(clean) + ind] * noiseratio
        # write wav
        noisywavname = noisy_splitdir + cleanwav[:-4] + "_" + noisewav[:-4] + "_snr" + str(snr) + ".wav"
        sf.write(noisywavname, noisyAudio, 8000)
    else:
        print("fs of clean and noise is unequal or the length of clean is longer than noise's\n")
        sys.exit(-1)

if __name__ == "__main__":
    noisedir = "E:\\gzcm\\Dataset\\noisex92"
    cleandir ="E:\\Dataset\\TIMIT\\TEST"
    # noisydir=

