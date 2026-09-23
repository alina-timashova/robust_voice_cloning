# necessary packages 
import io
import os
import csv
import time
import torch
import subprocess
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio 
import hashlib
import librosa
from tqdm import tqdm
from multiprocessing import Pool

# changeable function parameters - adjust to whichever specific dataset you are using
# hf = huggingface
DATASET_CONFIG = {'hf_path': 'ajaykarthick/codecfake-audio', 
                  'hf_split': 'train', 
                  'streaming': True,

                  # setting parameters for column names in the dataset 
                  'id_col': 'audio_id',
                  'audio_col': 'audio',
                  'label_col': 'real_or_fake',

                  # setting 'real' audio labels (0); anything else (false audios) are assigned a value of 1
                  'real_label_values': {'R'},
                  'keep_codecs': {'R', 'F02', 'F04'}, # F02 SpeechTokenizer, F04 EnCodec

                  # splitting
                  'seed': 42,
                  'split_ratios': {'train': 0.7, 'eval': 0.2, 'test': 0.1},

                  # audio
                  'target_sr': 16000,
                  'opus_bitrate': '16k',

                  # sample number, None if all
                  'max_clips': None,

                  # number of worker processes for parallelism
                  'n_workers': 4,

                  # speaker filtering
                  # 'p225_002' -> 'p225'  (speaker is everything before the first underscore)
                  'group': 'adult',
                  'age': '18+',
                  'speaker_filtering': '_',
                  'out_root': os.path.join(os.path.expanduser('~'), 'nn-dataprep', 'codecfake_preprocessed')
                  } 

# routing to the metadata file 
# clip_uid makes each row unique
METADATA_HEADER = ['clip_uid', 'group', 'age', 'audio_id', 'speaker', 'label', 'label_raw', 
                   'clean_path', 'compressed_path', 
                   'clean_sr', 'clean_n_samples', 'compressed_sr', 'compressed_n_samples'] 

def speaker_of(audio_id):
    '''
    Speaker filtering
    'p225_002' -> 'p225'  (speaker is everything before the first underscore)
    '''
    return audio_id.split(DATASET_CONFIG['speaker_filtering'])[0]

def keep_samples(label_raw, ds_cfg):
    '''
    Keep samples that are real (R) or encoded by target codecs (SpeechTokenizer, EnCodec)
    Return True to process clip, False to skip it
    '''
    return label_raw in ds_cfg['keep_codecs']

def assign_split(speaker, ratios):
    '''
    Splitting that works with streaming=True: all unique speakers are split
    into buckets based on given ratios

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
 
def label_classification(label_raw):
    '''
    Mapping a raw value back to the binary classification system (real audio get label 0, false gets label 1)
    '''
    return 0 if label_raw in DATASET_CONFIG['real_label_values'] else 1

 
def to_16k(a, sr, target_sr):
    '''
    Function to force the source clips ('clean' audio) into 16khz 
    so that model learns qualitative difference between real and fake
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
    # '-b:a', bitrate -> set how many bits per second comressed audio gets
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

def load_ds(ds_cfg):
    ds = load_dataset(ds_cfg['hf_path'], split=ds_cfg['hf_split'], streaming=ds_cfg['streaming'])
    ds = ds.cast_column(ds_cfg['audio_col'], Audio(decode=False))

    if ds_cfg['max_clips'] is not None:
        ds = ds.shuffle(seed=ds_cfg['seed'], buffer_size=10000)
    return ds

def split_outputs(ds_cfg):
    '''
    Creating the output folders and files, one for each split. We keep three things: 
    writers[split], the csv writer for that split's metadata
    meta_files[split], the open file handle 
    done[split],the set of audio_ids already processed 
    '''

    os.makedirs(ds_cfg['out_root'], exist_ok=True)
    writers, meta_files, done = {}, {}, {}

    for split in ('train', 'eval', 'test'):

        # creatng the clean and compressed audio folders for this split 
        for kind in ('clean', 'compressed'):
            os.makedirs(os.path.join(ds_cfg['out_root'], split, kind, 'audio'), exist_ok = True)

        # route this to the metadata file 
        meta_path = os.path.join(ds_cfg['out_root'], split, 'metadata.csv')

        # if a metadata file already exists from a previous run, then it will read back every audio_id it contains 
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

def process_one(args):
    '''
    Worker function: the code that each worker runs on a single clip in parallel
    Receive one clip -> do decoding -> do clean and comressed version of clip -> return finished audio

    Instead of writing each clip to the file, does heavy work and returns the result, so the main
    function collects them and write to the file collectively
    '''
    sample, ds_cfg = args # unpacks tuple into two variables, one clip's data and config dict
                            # tuple is handed by imap_unordered in process_dataset()
    audio_id = sample[ds_cfg['id_col']]
    label_raw = sample[ds_cfg['label_col']]

    try:
        array, sr = sf.read(io.BytesIO(sample[ds_cfg['audio_col']]['bytes']))
        clean, clean_sr, compressed, compressed_sr = clip_versions(array, sr, ds_cfg) # <- main task we are parallelizing

        return {
            'audio_id': audio_id,
            'label_raw': label_raw,                    
            'clean': clean,
            'clean_sr': clean_sr,
            'compressed': compressed,
            'compressed_sr': compressed_sr,
        }
    except Exception as e:
        return {'audio_id': audio_id, 'error': str(e)}

def process_dataset(ds, ds_cfg):
    '''
    Main pipeline: transform each clip into clean & comressed version
    and save them and their metadata to corresponding file
    '''

    processed = {'train': 0, 'eval': 0, 'test': 0}
    total = skipped = failed = 0

    # recording start time for tracking purposes 
    start = time.time()

    writers, meta_files, done = split_outputs(ds_cfg)
    n_workers = ds_cfg.get('n_workers', 1)

    def wanted():
        '''
        Generator function: instead of processing the whole list of clips,
        loops over list of clips lazily, i.e. hands one object at a time when worker asks for one

        return vs yield:
            return -> runs top to bottom, produces the entire list first and only then returns, done
            yield -> hands one item out and then pauses, freezing at exact state (position in for loop with all its variables);
                    resumes, when smth asks for the next item and then pauses again
        '''
        nonlocal skipped # permission to modify outer 'skipped' counter that is not part of inner function
        for sample in ds:
            audio_id = sample[ds_cfg['id_col']]
            label_raw = sample[ds_cfg['label_col']]

            if not keep_samples(label_raw, ds_cfg):
                continue

            clip_uid = f'{audio_id}__{label_raw}'
            split = assign_split(speaker_of(audio_id), ds_cfg['split_ratios'])
    
            # if this clip is already in the split's metadata from a previous run
            # this tells the function to skip it 
            if clip_uid in done[split]:
                skipped += 1
                continue

            yield (sample, ds_cfg)

    with Pool(n_workers) as pool: # Pool creates a group of worker processes and distributes tasks among them
                                    # with n_workers=4 -> starts 4 separate worker processes, each a full independent Python running on its own CPU core

        # pool clip from wanted(), run process_one() on each across the workers
        # chunksize=8 -> each worker processes 8 clips one by one, then requests next batch of 8
        # imap_unordered -> results arrive at whatever order they are completed (doesn't matter since each clip has unique ID)
        for clip_result in tqdm(pool.imap_unordered(process_one, wanted(), chunksize=8), unit='clip'): # clip_result = dict returned by process_one()
            if 'error' in clip_result:
                failed += 1
                print(f"failed {clip_result['audio_id']}: {clip_result['error']}")
                continue

            audio_id = clip_result['audio_id']
            label_raw = clip_result['label_raw']
            clip_uid = f'{audio_id}__{label_raw}'
            split = assign_split(speaker_of(audio_id), ds_cfg['split_ratios'])
            label = label_classification(label_raw)
            safe_id = clip_uid.replace('/', '_').replace('\\', '_')
            clean_path = os.path.join(ds_cfg['out_root'], split, 'clean', 'audio', safe_id + '.npy')
            comp_path  = os.path.join(ds_cfg['out_root'], split, 'compressed', 'audio', safe_id + '.npy')
            np.save(clean_path, clip_result['clean'].astype(np.float32))
            np.save(comp_path, clip_result['compressed'].astype(np.float32))

            writers[split].writerow([clip_uid, ds_cfg['group'], ds_cfg['age'], audio_id, speaker_of(audio_id), label, label_raw,
                                                clean_path, comp_path, 
                                                clip_result['clean_sr'], clip_result['clean'].shape[0],
                                                clip_result['compressed_sr'], clip_result['compressed'].shape[0]])

            processed[split] += 1
            total += 1 
            
            # force metadata to be physically saved to disk after every 100 clips
            if total % 100 == 0:
                meta_files[split].flush()

            if total % 500 == 0:
                # clips per second
                rate = total / (time.time() - start)
    
                # calculation possible for any values (defined in the CONFIG section)
                if ds_cfg['max_clips']:
                    remaining = (ds_cfg['max_clips'] - total) / rate / 60 if rate else 0
                    print(f'{total} done | {rate:.1f} clips | ~{remaining:.0f}')
    
                else:
                    print(f'{total} done | {rate:.1f} clips/s')

            if ds_cfg['max_clips'] is not None and total >= ds_cfg['max_clips']:
                break

    # closing all files once the loop is finished
    for mf in meta_files.values():
        mf.close()

    # final summary 
    print(f'total time {(time.time() - start) / 60:.1f} min: '
          f'{processed} processed, {skipped} skipped, {failed} failed')

def mel_transform(audio, target_frames=126):

  mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80)
  mel_db = librosa.power_to_db(mel, ref=np.max) # normalize

  # Crop if too long
  if mel_db.shape[1] > target_frames:
    mel_db = mel_db[:, :target_frames]

  # Pad if too short
  elif mel_db.shape[1] < target_frames:
    padding = target_frames - mel_db.shape[1]
    mel_db = np.pad(
        mel_db,
        ((0, 0), (0, padding)),
        mode='constant',
        constant_values=mel_db.min()
    )

  return mel_db.astype(np.float32)

def mel_to_tensor(mel):
    # NumPy array: (80, 126)
    tensor = torch.from_numpy(mel).float()

    # Add channel dimension: (1, 80, 126)
    tensor = tensor.unsqueeze(1)

    return tensor


def load_split(split, version, out_root, transform=None):
    '''
    FOR NEURAL MODEL 

    Load one split's audio & labels

    split: 'train', 'eval', 'test'
    version: 'clean', 'compressed'
    out_root: preprocessing output root
    transform: for Librosa
    '''

    if version not in ('clean', 'compressed'):
        raise ValueError(f"version must be 'clean' or 'compressed', got {version}")

    meta_path = os.path.join(out_root, split, 'metadata.csv')
    X, y, ids = [], [], []
    with open(meta_path, newline='') as f:
        for row in csv.DictReader(f):
            audio = np.load(row[f'{version}_path'])
            if transform is not None:
                audio = transform(audio) # Librosa transformation goes here
            X.append(audio)
            y.append(int(row['label']))
            ids.append(row['clip_uid'])

    return X, y, ids # keep ids to backtrace misclassification, e.g. can listen the samples that model got wrong

    # example use
    # X_train, y_train, _ = load_split('train', 'clean', OUT_ROOT) <- you get original CodecFake samples without compression for training

def main():
    ds = load_ds(DATASET_CONFIG)
    process_dataset(ds, DATASET_CONFIG)

if __name__ == '__main__':
    main()
