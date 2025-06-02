import numpy as np
import librosa
import os
import soundfile as sf

# 杨毅杰的语音分段
# def wav_split(wav, win_length, strid):
#     slices = []
#     if len(wav) > win_length:
#
#         for idx_end in range(win_length, len(wav), strid):
#             idx_start = idx_end - win_length
#             slice_wav = wav[idx_start:idx_end]
#             slices.append(slice_wav)
#
#         # 拼接最后一帧
#         slices.append(wav[-win_length:])
#     return slices
#
# #分段语音保存
# def save_slices(slices, name):
#     name_list = []
#     if len(slices) > 0:
#         for i, slice_wav in enumerate(slices):
#             name_slice = name + "_" + str(i) + '.npy'
#             np.save(name_slice, slice_wav)
#             name_list.append(name_slice)
#     return name_list

def wav_split(wav, win_length, strid):
    slices = []
    if len(wav) > win_length:

        for idx_end in range(win_length, len(wav), strid):
            idx_start = idx_end - win_length
            slices_wav = wav[idx_start:idx_end]
            # print("clean_slices", slices_wav.shape)
            slices.append(slices_wav)

    else:
            slices_wav = np.zeros((win_length), dtype=float)
            slices_wav[0:len(wav)] = wav
            slices.append(slices_wav)
            # 拼接最后一帧
            slices.append(wav[-win_length:])
    return slices


def save_slices(slices, name):
    name_list = []
    if len(slices) > 0:
        for i, slice_wav in enumerate(slices):
            name_slice = name + "_" + str(i) + '.wav'
            sf.write(name_slice, slice_wav,sr)
            name_list.append(name_slice)
            print("namalist:",name_list)


    return name_list



if __name__ == "__main__":
    clean_wav_path = r"/root/autodl-tmp/gzcm/BodSpeBD/BodSpeeDB/"
    # noise_wav_path = r"E:\gzcm\Dataset\voicebank\DS_10283_1942\noisy_testset_wav"

    catch_train_clean = r'/root/autodl-tmp/gzcm/BodSpeBD/wavesplit/clean/'
    # catch_train_noise = r'E:\gzcm\Dataset\voicebank\voicebank_wavesplit\test_noisy'

    os.makedirs(catch_train_clean, exist_ok=True)
    # os.makedirs(catch_train_noise, exist_ok=True)

    win_length = 65536
    strid = int(win_length / 2)
    # 遍历所有wav文件

    for root, dirs, files in os.walk(clean_wav_path):
        for file in files:
            file_clean_name = os.path.join(root, file)
            name = os.path.split(file_clean_name)[-1]
            if name.endswith("wav"):
                print("processing file %s" % (file_clean_name))
                clean_data,sr=librosa.load(file_clean_name,sr=16000,mono=True)
                clean_slices = wav_split(clean_data, win_length, strid)
                clean_namelist = save_slices(clean_slices, os.path.join(catch_train_clean, name))
                print(clean_data.shape)
                # print(clean_namelist)

                # 干净语音分段+保存
                clean_slices = wav_split(clean_data, win_length, strid)
                clean_namelist = save_slices(clean_slices, os.path.join(catch_train_clean, name))

    # for root, dirs, files in os.walk(noise_wav_path):
    #     for file in files:
    #         file_noise_name = os.path.join(root, file)
    #         name = os.path.split(file_noise_name)[-1]
    #         print(name)
    #         if name.endswith("wav"):
    #             print("processing file %s" % (file_noise_name))
    #             noisy_data, sr = librosa.load(file_noise_name, sr=16000, mono=True)
    #             print(noisy_data.shape)

                # 噪声语音分段+保存
                # noise_slices = wav_split(noisy_data, win_length, strid)
                # noise_namelist = save_slices(noise_slices, os.path.join(catch_train_noise, name))






# if __name__ == "__main__":
#     clean_wav_path = r"E:\gzcm\Dataset\thchs\thchs_wavesplit\Train\clean"
#     # noise_wav_path = r"E:\gzcm\Dataset\thchs\thchs_wavesplit\Train\noisy"
#
#     catch_train_clean = r'G:\gzcm\Dataset\thchs\thchs_wavesplit\clean'
#     # catch_train_noise = r'G:\gzcm\Dataset\thchs\thchs_wavesplit\noisy'
#
#     os.makedirs(catch_train_clean, exist_ok=True)
#     # os.makedirs(catch_train_noise, exist_ok=True)
#
#     win_length = 65536
#     strid = int(win_length / 2)
#     # 遍历所有wav文件
#     with open("E:\\gzcm\\SEModel\\train_data.scp", 'wt') as f:
#         for root, dirs, files in os.walk(noise_wav_path):
#             for file in files:
#                 file_noise_name = os.path.join(root, file)
#                 name = os.path.split(file_noise_name)[-1]
#                 if name.endswith("wav"):
#
#                     file_clean_name = os.path.join(clean_wav_path, name)
#                     print("processing file %s" % (file_clean_name))
#
#                     if not os.path.exists(file_clean_name):
#                         print("can not find file %s" % (file_clean_name))
#                         continue
#                     clean_data,sr=librosa.load(file_clean_name,sr=16000,mono=True)
#
#                     noisy_data, sr = librosa.load(file_noise_name, sr=16000, mono=True)
#                     print(clean_data.shape)
#                     print(noisy_data.shape)
#                     if not len(clean_data) == len(noisy_data):
#                         print("file length are not equal")
#                         continue
#                     # 干净语音分段+保存
#                     clean_slices = wav_split(clean_data, win_length, strid)
#                     clean_namelist = save_slices(clean_slices, os.path.join(catch_train_clean, name))
#
#                     # # 噪声语音分段+保存
#                     noise_slices = wav_split(noisy_data, win_length, strid)
#                     noise_namelist = save_slices(noise_slices, os.path.join(catch_train_noise, name))
#
#                     for clean_catch_name, noise_catch_name in zip(clean_namelist, noise_namelist):
#                         f.write("%s %s\n" % (clean_catch_name, noise_catch_name))
#



