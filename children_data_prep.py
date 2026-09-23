# necessary packages
import io
import os
import csv
import time
import subprocess
import random
import numpy as np
import soundfile as sf
import torch          # needed for codec models and torchaudio decoding
from datasets import load_dataset, Audio
from huggingface_hub import snapshot_download  # downloads the SpeechTokenizer checkpoint
from encodec import EncodecModel                # EnCodec (F04)
from encodec.utils import convert_audio         # EnCodec (F04)
from speechtokenizer import SpeechTokenizer     # SpeechTokenizer (F02)
import hashlib
import librosa
from tqdm import tqdm

# Samromur Children (language-and-voice-lab/samromur_children on HF) is a
# corpus containing only REAL children's speech; there are no fake speech samples here
# major changes from the intial preprocessing script include:
#   - no real_or_fake column to read or filter on
#   - every kept clip gets label 0 (real), fakes are generated as part of pipeline
#   - the dataset already provides a 'speaker_id' column, so we no longer need to include this
#     to parse the speaker out of the audio_id the way CodecFake does

# since there are no fake samples at the source, each real clip is now also run
# through SpeechTokenizer (F02) and EnCodec (F04), the same codec labels CodecFake uses,
# so every split ends up with a real+fake mix instead of a real-only one.
# this pipeline is separate from preprocessing_additional_codecs.py, which builds a DAC/EnCodec/SpeechTokenizer
# test-only set for unseen-codec generalization and is not reused here.

# device needed to load the codec models onto GPU/CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# age bins used to report the corresponding age category of the speaker
AGE_BINS = [(0, 9, '0-9'),
            (10, 12, '10-12'),
            (13, 17, '13-17')]

DATASET_CONFIG = {'hf_path': 'language-and-voice-lab/samromur_children',
                  'hf_split': 'train',
                  'streaming': True,

                  # column names in the dataset
                  'id_col': 'audio_id',
                  'audio_col': 'audio',
                  'speaker_col': 'speaker_id',
                  'age_col': 'age', # per-sample self-reported age

                  # splitting
                  'seed': 42,
                  'split_ratios': {'train': 0.7, 'eval': 0.2, 'test': 0.1},

                  # audio
                  'target_sr': 16000,
                  'opus_bitrate': '16k',

                  # sample number, None if all
                  'max_clips': None,

                  'codec_batch_size': 16,
                  'bucket_batches': 8,

                  'group': 'child',
                  'age_fallback': 'unknown', # used if a row is missing age_col
                  'age_bins': AGE_BINS,

                  # replaces the old max_speakers cap with an upfront, age-balanced
                  # pick of N speakers (N = number of unique adult CodecFake speakers), so
                  # both groups have matching speaker counts. None = keep every speaker
                  'match_adult_speaker_count': True,
                  'num_speakers': None,
                  'adult_out_root': os.path.join(os.path.expanduser('~/nn-dataprep'), 'codecfake_preprocessed'),

                  'out_root': os.path.join(os.path.expanduser('~/nn-dataprep'), 'samromur_preprocessed_sp')
                  }

# routing to the metadata file
# clip_uid makes each row unique
METADATA_HEADER = ['clip_uid', 'group', 'age', 'audio_id', 'speaker', 'label', 'label_raw',
                   'clean_path', 'compressed_path',
                   'clean_sr', 'clean_n_samples', 'compressed_sr', 'compressed_n_samples']

def pad_and_stack(waveform, num_channels):

    lengths = [w.shape[-1] for w in waveform]
    max_len = max(lengths)
    batch = torch.zeros(len(waveform), num_channels, max_len)
    for i, w in enumerate(waveform):
        batch[i, :, :w.shape[-1]] = w
    return batch, lengths

def build_speechtokenizer(device):
    '''
    Generate fakes with SpeechTokenizer
    '''

    path = snapshot_download(
        repo_id='fnlp/SpeechTokenizer',
        allow_patterns=['speechtokenizer_hubert_avg/*'],
    )

    ckpt_dir = os.path.join(path, 'speechtokenizer_hubert_avg')
    model = SpeechTokenizer.load_from_checkpoint(
        os.path.join(ckpt_dir, 'config.json'),
        os.path.join(ckpt_dir, 'SpeechTokenizer.pt'),
    ).to(device)

    model.eval()
    model_sr = model.sample_rate

    def process_batch(waveforms, sample_rates):
        '''
        To accelerate pipeline running on cluster GPU,
        transform clips in batches

        Returns a list of (numpy array, sample rate)
        with padding removed
        '''
        model_wavs = [convert_audio(wav, sr, model_sr, 1)
                      for wav, sr in zip(waveforms, sample_rates)]
        batch, lengths = pad_and_stack(model_wavs, num_channels=1)
        batch = batch.to(device)

        with torch.no_grad():
            codes = model.encode(batch)
            reconstructed = model.decode(codes)

        reconstructed = reconstructed.cpu()
        # safety check in case model returns channel dimension, need to guarantee 2D output
        if reconstructed.dim() == 3:
            reconstructed = reconstructed.squeeze(1)

        return [(reconstructed[i, :lengths[i]].numpy(), model_sr)
                for i in range(len(model_wavs))]

    return process_batch

def build_encodec(device, bandwidth=6.0):
    '''
    Generate fakes with EnCodec
    '''

    model = EncodecModel.encodec_model_24khz().to(device)
    model.set_target_bandwidth(bandwidth)
    model.eval()
    model_sr = model.sample_rate
    num_channels = model.channels

    def process_batch(waveforms, sample_rates):
        model_wavs = [convert_audio(wav, sr, model_sr, num_channels)
                      for wav, sr in zip(waveforms, sample_rates)]

        batch, lengths = pad_and_stack(model_wavs, num_channels)
        batch = batch.to(device)

        with torch.no_grad():
            encoded = model.encode(batch)
            reconstructed = model.decode(encoded)

        reconstructed = reconstructed.cpu()
        if reconstructed.dim() == 3:
            reconstructed = reconstructed.squeeze(1)
        
        return [(reconstructed[i, :lengths[i]].numpy(), model_sr)
            for i in range(len(model_wavs))]

    return process_batch

def build_codecs(device):
    '''
    Map each fake label to its function
    '''
    return {
        'F02': build_speechtokenizer(device),
        'F04': build_encodec(device),
    }

def age_to_number(raw_age):
    '''
    Check the proper formatting of age feature and
    convert the raw age value to a number
    '''
    if raw_age is None:
        return None
    text = str(raw_age).strip()
    if text == '':
        return None

    if '-' in text:
        low, high = text.split('-')
        try:
            return (int(low) + int(high)) / 2 # if age is a range like 13-17, return midpoint
        except ValueError:
            return None

    try:
        return float(text)
    except ValueError:
        return None

def age_group(raw_age, bins):
    '''
    Return the label of the age bin the raw age falls to or 'unknown'
    '''
    age_number = age_to_number(raw_age)
    if age_number is None:
        return 'unknown'
    for low, high, label in bins:
        if low <= age_number <= high:
            return label

    return 'unknown'


def speaker_of(sample, ds_cfg):
    '''
    Samromur already provides a speaker_id column, so we use it directly
    (unlike CodecFake, where the speaker has to be parsed out of audio_id).
    '''
    return sample[ds_cfg['speaker_col']]

def assign_split(speaker, ratios):
    '''
    Splitting that works with streaming = True
    Ensures that each speaker is only seen in one split
    '''
    bucket = int(hashlib.md5(speaker.encode()).hexdigest(), 16) % 100
    train_end = ratios['train'] * 100
    eval_end = train_end + ratios['eval'] * 100
    if bucket < train_end:
        return 'train'
    if bucket < eval_end:
        return 'eval'
    return 'test'

def count_adult_speakers(adult_out_root):
    '''
    Count unique speakers across the train/eval/test metadata
    of the already-processed adult data
    '''
    speakers = set()
    for split in ('train', 'eval', 'test'):
        meta_path = os.path.join(adult_out_root, split, 'metadata.csv')
        if os.path.exists(meta_path):
            with open(meta_path, newline='') as f:
                speakers.update(row['speaker'] for row in csv.DictReader(f))

    return len(speakers)

def select_speakers(ds_cfg, target_count):
    '''
    Replaces the old incremental max_speakers cap. Runs a lightweight, audio-free
    pass over the dataset to map each speaker to an age, then picks num_speakers speakers,
    spread across age groups as evenly as the dataset allows, so the child speaker count
    can be matched to the adult speaker count.
    '''
    if target_count is None:
        return None
    
    meta_ds = load_ds(ds_cfg, columns=[ds_cfg['id_col'],
                                       ds_cfg['speaker_col'],
                                       ds_cfg['age_col']])
    speaker_group = {}
    for row in meta_ds:
        speaker = row[ds_cfg['speaker_col']]
        speaker_group.setdefault(speaker, age_group(row.get(ds_cfg['age_col']), ds_cfg['age_bins']))

    speakers = list(speaker_group)
    if target_count >= len(speakers):
        print(f'NOTE: asked for {target_count} child speakers but only {len(speakers)} exists -> keeping all')
        return set(speakers)

    rng = random.Random(ds_cfg['seed'])
    chosen = set(rng.sample(speakers, target_count))

    achieved = {}
    for speaker in chosen:
        label = speaker_group[speaker]
        achieved[label] = achieved.get(label, 0) + 1

    print(f'selected {len(chosen)} child speakers by age group: {achieved}')

    return chosen

def to_16k(a, sr, target_sr):
    '''
    Function to force the source clips ('clean' audio) into 16khz
    so that model learns qualitativee difference between real and fake
    Forces an array to 16 kHz mono float32

    a = audio array
    sr = current rate
    target_sr = desired rate (16 kHz)
    '''

    # if the clip is multi-channel (for example, a 2D array)
    if a.ndim > 1:

        # force the channels into an average of 1
        a = a.mean(axis = 1)

    # if the sample rate is not already 16khz
    if sr != target_sr:

        # convert it
        a = librosa.resample(a, orig_sr=sr, target_sr=target_sr)

    # returns the converted value
    return a.astype(np.float32)


def opus_roundtrip(array, sr, bitrate, target_sr):
    '''
    Encode array to Opus and decode it back, returns target_sr result

    sr = sample rate 
    '''

    # write the raw array into an in-memory WAV buffer (no temp file)
    wav_buf = io.BytesIO() # empty in-memory binary buffer
    sf.write(wav_buf, array, sr, format='WAV', subtype='PCM_16') # encode array into in-memory buffer
    wav_bytes = wav_buf.getvalue() # extracts everything written into buffer as a syngle bytes object

    # everything before '-i' decribes inputs: 
    # 'ffmpeg' -> program to run
    # '-y' -> overwriting (yes), harmless here bc nothing to overwrite
    # '-f', 'wav' -> force input format to .wav
    # '-i', 'pipe:0' -> read the audio from the bytes we pipe in instead of a file (stdin, file descriptor 0)
    # after '-i' is output:
    # '-c:a', 'libopus' -> codec for audio = libopus (Opus encoder)
    # '-b:a', bitrate -> set how many bits per second compressed audio gets
    # '-ac', '1' -> number of audio channels
    # '-application', 'voip' -> Opus-specific tuning mode, 'voip' optimizes for speech
    # '-f', 'ogg' -> force output format to .ogg
    # 'pipe:1' -> outout to stdout (file descriptor 1)

    # stdout=subprocess.PIPE -> catches output so that Python can read it, without it goes to terminal and is lost
    # stderr=subprocess.PIPE -> catches stderr (errors and diagnostic) + keeps logs clean
    # check=True -> raises and expection if ffmpeg fails

    encode =  subprocess.run(['ffmpeg', '-y', '-f', 'wav', '-i', 'pipe:0',
                              '-c:a', 'libopus', '-b:a', bitrate,
                              '-ac', '1', '-application', 'voip',
                              '-f', 'ogg', 'pipe:1'],
                              input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,)
    opus_bytes = encode.stdout

    # '-ar', str(target_sr) -> resample decoded audio to target sample rate (16 kHz)
    decode = subprocess.run(
        ['ffmpeg', '-y', '-f', 'ogg', '-i', 'pipe:0',
        '-ar', str(target_sr), '-f', 'wav', 'pipe:1'],
        input=opus_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )

    out, out_sr = sf.read(io.BytesIO(decode.stdout))

    return out.astype(np.float32), out_sr

def clip_versions(array, sr, ds_cfg):
    '''
    Turn clip into two versions:
        1. Clean = original clip that is guaranteed to be 16kHz and mono
        2. Compressed = undergoes Opus compression
    '''

    clean = to_16k(array, sr, ds_cfg['target_sr'])
    compressed, comp_sr = opus_roundtrip(
        clean, ds_cfg['target_sr'], ds_cfg['opus_bitrate'], ds_cfg['target_sr']
    )

    return clean, ds_cfg['target_sr'], compressed, comp_sr

def decode_to_tensor(raw_bytes):
    '''
    Takes raw_bytes and turns into mono tensor (1, time_steps) + sample rate.
    Uses soundfile (not torchaudio) to avoid the TorchCodec backend.
    '''
    audio, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype='float32')
    waveform = torch.from_numpy(audio)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.transpose(0, 1)
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, sample_rate

def load_ds(ds_cfg, columns=None):
    '''
    Load the streaming dataset
    '''
    ds = load_dataset(ds_cfg['hf_path'], split=ds_cfg['hf_split'], streaming=ds_cfg['streaming'])

    # optional column selection, used by select_speakers() below so the
    # speaker/age scan doesn't have to decode audio bytes it doesn't need
    if columns is not None:
        ds = ds.select_columns(columns)
    else:
        ds = ds.cast_column(ds_cfg['audio_col'], Audio(decode=False))

    if ds_cfg['max_clips'] is not None:
        ds = ds.shuffle(seed=ds_cfg['seed'], buffer_size=10000)
    return ds

def split_outputs(ds_cfg):
    '''
    Creating the output folders and files, one for each split. We keep three things:
    writers[split], the csv writer for that split's metadata
    meta_files[split], the open file handle
    done[split],the set of clip_ids already processed
    '''

    os.makedirs(ds_cfg['out_root'], exist_ok=True)
    writers, meta_files, done = {}, {}, {}

    for split in ('train', 'eval', 'test'):

        # creatng the clean and compressed audio folders for this split
        for kind in ('clean', 'compressed'):
            os.makedirs(os.path.join(ds_cfg['out_root'], split, kind, 'audio'), exist_ok = True)

        # route this to the metadata file
        meta_path = os.path.join(ds_cfg['out_root'], split, 'metadata.csv')

        # if a metadata file already exists from a previous run, then it will read back every clip_uid it contains
        # we can then skip those clips later instead of reprocessing them, starting empty if there's no file yet
        done[split] = set()
        if os.path.exists(meta_path):
            with open(meta_path, newline = '') as f:
                done[split] = {row['clip_uid'] for row in csv.DictReader(f)}

        # note whether the file is new before we open it
        new_file = not os.path.exists(meta_path)

        # opening the metadata file in append mode ('a') so a re-run adds to it (prevents rewriting previous results)
        meta_files[split] = open(meta_path, 'a', newline = '')
        writers[split] = csv.writer(meta_files[split])

        # headers for new files
        if new_file:
            writers[split].writerow(METADATA_HEADER)

    return writers, meta_files, done

def process_dataset(ds, ds_cfg, chosen_speakers, codecs):
    '''
    Main loop:
        processed = source clips per split with at least one new version written
        accepted = source clips decoded and queued for the codecs
        written = metadata rows written (real + fake versions)
        skipped = source clips whose versions we already written in previous run
        failed = clips that could not be decoded 
    '''

    processed = {'train': 0, 'eval': 0, 'test': 0}
    accepted = skipped = failed = written = 0

    # recording start time for tracking purposes
    start = time.time()

    writers, meta_files, done = split_outputs(ds_cfg)

    codec_batch_size = ds_cfg['codec_batch_size']
    buffer_limit = codec_batch_size * ds_cfg.get('bucket_batches', 8)
    buffer = []

    def write_versions(clip, version_list):
        '''
        Save clean + comressed audio and a metadata row for each version
        '''
        nonlocal written
        split = clip['split']
        wrote_any = False
        for label_raw, array, version_sr in version_list:
            clip_uid = f"{clip['audio_id']}__{label_raw}"
            if clip_uid in done[split]:
                continue
            label = 0 if label_raw == 'R' else 1
            clean, clean_sr, compressed, compressed_sr = clip_versions(array, version_sr, ds_cfg)
            safe_id = clip_uid.replace('/', '_').replace('\\', '_')
            clean_path = os.path.join(ds_cfg['out_root'], split, 'clean', 'audio', safe_id + '.npy')
            comp_path = os.path.join(ds_cfg['out_root'], split, 'compressed', 'audio', safe_id + '.npy')
            np.save(clean_path, clean.astype(np.float32))
            np.save(comp_path, compressed.astype(np.float32))
            writers[split].writerow([clip_uid, ds_cfg['group'], clip['age'], clip['audio_id'], clip['speaker'],
                                        label, label_raw, clean_path, comp_path,
                                        clean_sr, clean.shape[0], compressed_sr, compressed.shape[0]])
            done[split].add(clip_uid)
            written += 1
            wrote_any = True
        return wrote_any

    def flush():
        nonlocal failed
        if not buffer:
            return
        buffer.sort(key=lambda c: c['waveform'].shape[-1])   # length-bucketing
        for batch_start in range(0, len(buffer), codec_batch_size):
            micro = buffer[batch_start:batch_start + codec_batch_size]
            waveforms = [c['waveform'] for c in micro]
            sample_rates = [c['sample_rate'] for c in micro]
            try:
                fake_outputs = {label_raw: codec(waveforms, sample_rates)
                                for label_raw, codec in codecs.items()}
            except Exception as e:
                for clip in micro:
                    failed += 1
                    print(f"codec batch failed near {clip['audio_id']}: {e}")
                continue
            for i, clip in enumerate(micro):
                versions = [('R', clip['waveform'].squeeze(0).numpy(), clip['sample_rate'])]
                for label_raw in codecs:
                    fake_array, fake_sr = fake_outputs[label_raw][i]
                    versions.append((label_raw, fake_array, fake_sr))
                try:
                    if write_versions(clip, versions):
                        processed[clip['split']] += 1
                except Exception as e:
                    failed += 1
                    print(f"write failed {clip['audio_id']}: {e}")
        for mf in meta_files.values():
            mf.flush()
        buffer.clear()

    pbar = tqdm(ds, total=ds_cfg['max_clips'], unit='clip')
    for sample in pbar:

        if ds_cfg['max_clips'] is not None and accepted >= ds_cfg['max_clips']:
            break

        audio_id = sample[ds_cfg['id_col']]
        speaker = speaker_of(sample, ds_cfg)

        # only keep clips from the pre-selected, age-balanced speaker pool
        if chosen_speakers is not None and speaker not in chosen_speakers:
            continue

        split = assign_split(speaker, ds_cfg['split_ratios'])

        # R/F02/F04 all share this clip's audio_id, so check up front whether
        # every version has already been written before decoding/running codecs again
        expected_uids = [f'{audio_id}__R'] + [f'{audio_id}__{codec}' for codec in codecs]
        if all(uid in done[split] for uid in expected_uids):
            skipped += 1
            continue

        # per-sample self-reported age, falling back to a placeholder if missing
        age_value = age_to_number(sample.get(ds_cfg['age_col']))
        if age_value is None:
            age_value = ds_cfg['age_fallback']

        # forces any unreadable/corrupt clips to skip to not break the script halfway through
        try:
            waveform, sample_rate = decode_to_tensor(sample[ds_cfg['audio_col']]['bytes'])
        except Exception as e:
            failed += 1
            print(f'failed {audio_id}: {e}')
            continue

        buffer.append({'audio_id': audio_id, 'speaker': speaker, 'age': age_value,
                       'split': split, 'waveform': waveform, 'sample_rate': sample_rate})
        accepted += 1

        if len(buffer) >= buffer_limit:
            flush()

        # prints progress and processing rate every 500 clips
        if accepted % 500 == 0:

            # clips per second
            rate = accepted / (time.time() - start)

            if ds_cfg['max_clips']:
                remaining = (ds_cfg['max_clips'] - accepted) / rate / 60 if rate else 0
                print(f'{accepted} done | {rate:.1f} clips | ~{remaining:.0f}')

            else:
                print(f'{accepted} done | {rate:.1f} clips/s')

    flush()
    # closing all files once the loop is finished
    for mf in meta_files.values():
        mf.close()

    # final summary 
    print(f'total time {(time.time() - start) / 60:.1f} min: '
        f'{processed} processed, {written} rows, {skipped} skipped, {failed} failed')

def main():
    codecs = build_codecs(device)
    target_count = None

    if DATASET_CONFIG.get('match_adult_speaker_count'):
        target_count = count_adult_speakers(DATASET_CONFIG['adult_out_root'])
        print(f'Matching child speakers to {target_count} adult speakers')

        if target_count == 0:
            print('No adult metadata found')

    elif DATASET_CONFIG.get('num_speakers') is not None:
        target_count = DATASET_CONFIG['num_speakers']

    chosen_speakers = select_speakers(DATASET_CONFIG, target_count)
    ds = load_ds(DATASET_CONFIG)
    process_dataset(ds, DATASET_CONFIG, chosen_speakers, codecs)

if __name__ == '__main__':
    main()
