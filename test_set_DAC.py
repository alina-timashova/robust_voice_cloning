import os, io, csv
import torch, torchaudio
import numpy as np
from datasets import load_dataset, Audio
import dac
from tqdm import tqdm

from data_prep import to_16k, opus_roundtrip, speaker_of, assign_split

TARGET_SR = 16000
OPUS_BITRATE = '16k'

DAC_METADATA_HEADER = ['clip_uid', 'group', 'age', 'audio_id', 'speaker', 'label', 'label_raw', 
                   'clean_path', 'compressed_path', 
                   'clean_sr', 'clean_n_samples', 'compressed_sr', 'compressed_n_samples']

RATIOS = {'train': 0.7, 'eval': 0.2, 'test': 0.1} 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def dac_codec(device, model_type='16kHz'):
    model_path = dac.utils.download(model_type=model_type)
    model = dac.DAC.load(model_path).to(device)
    model.eval()

    def process(waveform, sample_rate):
        if sample_rate != model.sample_rate:
            waveform =torchaudio.functional.resample(waveform, sample_rate, model.sample_rate)

        length = waveform.shape[-1] # DAC adds padding to the audio, which we need to trim later to recover original length
        x = waveform.unsqueeze(0).to(device) # add batch dim: (channel, time_steps) -> (1, channel, time_steps)

        with torch.no_grad():
            x = model.preprocess(x, model.sample_rate) # here happens right-padding to prevent boundary artifacts
                                                        # waveform is filled with zeroes to become exact multiple of hop factor
                                                        # hop factor is usually 0.25 * window_length (75% overlap)

            # === ENCODING STEP ===

            # z - quantized latent representation of audio, (1, D, frames)
            # codes - integer indices for codebook lookup
            # latents - pre-quantization vector projections, maybe useful for analysis later
            # skipped values _ - lossed, only needed for training the codec

            z, codes, latents, _, _ = model.encode(x) # here happens convolutional encoding + quantization
                                                        # (1, channel, time_steps) -> (1, D, frames) where D - model's latent dim, frames = time / hop length
                                                        # this new representation goes through RVQ as in EnCodec

            # === DECODING STEP ===
            reconstructed = model.decode(z)

        reconstructed = reconstructed[..., :length].squeeze(0).cpu() # trim padding to recover original length + drop batch dim
        return reconstructed, model.sample_rate

    return process

def decode_to_tensor(raw_bytes):
    '''
    Takes raw_bytes and turns into mono tensor (1, time_steps) + sample rate
    '''

    waveform, sample_rate = torchaudio.load(io.BytesIO(raw_bytes))

    if waveform.shape[0] > 1:
            print('Found audio with multiple channels, converting to mono')
            waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sample_rate

def clip_versions(array, sr):
    '''
    Turn clip into two versions:
        1. Clean = original clip that is guaranteed to be 16kHz and mono
        2. Compressed = undergoes Opus compression
    '''

    clean = to_16k(array, sr, TARGET_SR)
    compressed, comp_sr = opus_roundtrip(clean, TARGET_SR, OPUS_BITRATE, TARGET_SR)

    return clean, TARGET_SR, compressed, comp_sr

def write_row(writer, out_root, clip_uid, group, audio_id, speaker, age,
               label, label_raw, array_np, sr):

    clean, clean_sr, comp, comp_sr = clip_versions(array_np, sr)
    safe_id = clip_uid.replace('/', '_').replace('\\', '_')
    clean_path = os.path.join(out_root, 'test', 'clean', 'audio', safe_id + '.npy')
    comp_path  = os.path.join(out_root, 'test', 'compressed', 'audio', safe_id + '.npy')
    np.save(clean_path, clean.astype(np.float32))
    np.save(comp_path, comp.astype(np.float32))

    # recording one row in this split's metadata table describing the clip
    writer.writerow([clip_uid, group, age, audio_id, speaker, 
                     label, label_raw, clean_path, comp_path, 
                     clean_sr, clean.shape[0], comp_sr, comp.shape[0]])

def build_dac_testset(dataset, dac_process, group, out_root,
                      id_fn, speaker_fn, age_fn, keep_fn, max_samples=500):
    for kind in ('clean', 'compressed'):
          os.makedirs(os.path.join(out_root, 'test', kind, 'audio'), exist_ok=True)
    meta_path = os.path.join(out_root, 'test', 'metadata.csv')
    new_file = not os.path.exists(meta_path)
    f = open(meta_path, 'a', newline='')
    writer = csv.writer(f)
    if new_file:
         writer.writerow(DAC_METADATA_HEADER)

    n = 0
    for sample in tqdm(dataset):
        if max_samples is not None and n >= max_samples:
            break
        if not keep_fn(sample): # allows to choose only adults or only children
            continue
        audio_id, speaker, age = id_fn(sample), speaker_fn(sample), age_fn(sample)

        try:
            wav, sr = decode_to_tensor(sample['audio']['bytes'])

            # writer, out_root, clip_uid, group, audio_id, speaker, age, label, label_raw, array_np, sr
            write_row(writer, out_root, f'{group}__{audio_id}__real', group, audio_id,
                      speaker, age, 0, 'R', wav.squeeze(0).numpy(), sr)

            dac_wav, dac_sr = dac_process(wav, sr)
            write_row(writer, out_root, f'{group}__{audio_id}__dac', group, audio_id,
                                  speaker, age, 1, 'DAC', dac_wav.squeeze(0).numpy(), dac_sr)

            f.flush(); n += 1

        except Exception as e:
             print(f'Failed {audio_id}: {e}')
             continue

    f.close()
    print(f'{group}: {n} source clips -> {2*n} rows in {out_root}')

# ==== BUILDING TEST SETS ====

dac_process = dac_codec(device, model_type='16kHz')

# ==== ADULTS ====

codecfake = load_dataset("ajaykarthick/codecfake-audio", split='train', streaming=True)
codecfake = codecfake.cast_column("audio", Audio(decode=False))
codecfake = codecfake.shuffle(seed=42, buffer_size=10000)
build_dac_testset(
     codecfake, dac_process, group='adult',
     out_root='dac_test_adults',
     id_fn=lambda s: s['audio_id'],
     speaker_fn=lambda s: speaker_of(s['audio_id']),
     age_fn=lambda s: '18+',
     keep_fn=lambda s: s['real_or_fake'] == 'R' and assign_split(speaker_of(s['audio_id']), RATIOS) == 'test',
     max_samples=None,
     )
