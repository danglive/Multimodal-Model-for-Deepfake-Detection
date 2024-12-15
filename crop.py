import cv2
import os
from multiprocessing import Process, JoinableQueue
import traceback
import gc
import json
import warnings
import time
from retinaface.pre_trained_models import get_model
import numpy as np
from SyncNetModel import *
import librosa.display
import python_speech_features
from argparse import ArgumentParser
from utils import log, create_folder, calc_pdist
from scipy.ndimage.filters import uniform_filter1d
import librosa
import torch
from moviepy.editor import AudioFileClip, VideoFileClip, CompositeAudioClip
from moviepy.video.io import ImageSequenceClip
import torch
import matplotlib.pylab as plt

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

queue = JoinableQueue()
extraction_queue = JoinableQueue()
fail_queue = JoinableQueue()

def apply_crop(frame, size, y, x, margin=0.4):
    """
    Extract the face region from the frame using the bounding box center (x,y) and size.
    margin: to allow extra context around the face.
    """
    bs = size
    bsi = int(bs*(1+2*margin))
    frame = np.pad(frame, ((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = y+bsi
    mx  = x+bsi
    face = frame[int(my-bs):int(my+bs*(1+2*margin)), int(mx-bs*(1+margin)):int(mx+bs*(1+margin))]
    return face

def crop_faces(file_path, face_detector, frames, skip_num=4, crop_batch_size=8, crop_threshold=0.975):
    """
    Detect faces in frames using RetinaFace in batches.
    skip_num: we only detect on every skip_num-th frame to speed up inference.
    crop_threshold: minimum confidence for detection.
    """
    all_boxes, all_probs = [], []
    indices = range(0, len(frames), skip_num)
    select_frames = frames[indices, :, :, :]

    for i in range(int(np.ceil(len(select_frames)/crop_batch_size))):
        batch = select_frames[i*crop_batch_size: (i+1)*crop_batch_size]
        results = face_detector.predict_jsons_batch(batch)
        boxes = [[box["bbox"] for box in frame_result if box["score"] > crop_threshold] for frame_result in results]
        probs = [[box["score"] for box in frame_result if box["score"] > crop_threshold] for frame_result in results]

        all_boxes += boxes
        all_probs += probs

    # Interpolate face detection results for all frames
    return_boxes, return_probs = [], []
    for i in range(len(frames)):
        j = i // skip_num
        distance = 0
        boxes, probs = [], []
        while not boxes:
            l = max(0, j - distance)
            r = min(j + distance, len(all_probs)-1)
            boxes = all_boxes[l] or all_boxes[r]
            probs = all_probs[l] or all_probs[r]

            if l == 0 and r == len(all_probs)-1 and not boxes:
                log("No faces found. Use raw frames!", file_path)
                # If no face found at all, revert to whole frame crop
                h, w = frames[0].shape[:2]
                return_boxes = [[0, 0, w, h] for _ in range(len(frames))]
                return_probs = [1 for _ in range(len(frames))]
                return return_boxes, return_probs

            distance += 1
        return_boxes += [boxes[0]]
        return_probs += [probs[0]]
    return return_boxes, return_probs

def compute_latency(__S__, faces, audio, resample_rate, batch_size=8, vshift=15):
    """
    Compute audio-video sync latency using the SyncNet model.
    """
    with torch.no_grad():
        lastframe = len(faces) - 5
        faces = np.stack(faces, axis=3)
        faces = np.expand_dims(faces, axis=0)
        faces = np.transpose(faces, (0,3,4,1,2))  # (1, C, T, H, W)

        # Compute MFCC for audio
        mfcc = zip(*python_speech_features.mfcc(audio, resample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.from_numpy(cc.astype(float)).float()

        im_feat = []
        cc_feat = []
        # Extract features in batches
        for i in range(0, lastframe, batch_size):
            im_batch = [torch.tensor(faces[:, :, vframe:vframe+5, :, :]) for vframe in range(i, min(lastframe, i+batch_size))]
            im_in = torch.cat(im_batch, 0)
            im_out = __S__.forward_lip(im_in.cuda().half())
            im_feat.append(im_out.data.cpu())

            cc_batch = [cct[:, :, :, vframe*4:vframe*4+20] for vframe in range(i, min(lastframe, i+batch_size))]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = __S__.forward_aud(cc_in.cuda().half())
            cc_feat.append(cc_out.data.cpu())

        if len(im_feat) == 0 or len(cc_feat) == 0:
            # No latency computed if no frames
            return [], 0, 0

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
        mdist = torch.mean(torch.stack(dists, 1), 1).float()

        minval, minidx = torch.min(mdist, 0)
        offset = vshift - minidx
        conf = torch.median(mdist) - minval
        return mdist, offset.item(), conf.item()

def extract(save_image, frame_num, frame_size, output):
    """
    Extract faces and mel spectrograms and save them as .npz files.
    Runs in a separate process to handle I/O asynchronously.
    """
    while True:
        elem = extraction_queue.get()
        if elem == None:
            extraction_queue.task_done()
            return

        file_path, start, end, select_faces, mdist, latency, conf = elem
        file_name = file_path.split("/")[-1].split(".")[0]

        # Extract mel spectrograms for each frame chunk
        visible_folder = f"{output}/{file_name}"
        create_folder(visible_folder)
        mel_3cs = []
        if len(mdist) > 0:
            offset = 0
            duration = (end - start) / frame_num
            audioclip = AudioFileClip(file_path)
            audioclip.write_audiofile(f"{output}/{file_name}.wav", 44100, 2, 2000, "pcm_s32le")

            for i in range(frame_num):
                samples, sample_rate = librosa.load(f"{output}/{file_name}.wav", offset=offset, duration=duration)
                mel = librosa.power_to_db(librosa.feature.melspectrogram(y=samples, sr=sample_rate))
                offset += duration

                fig = plt.figure(figsize=[1,1])
                ax = fig.add_subplot(111)
                ax.axes.get_xaxis().set_visible(False)
                ax.axes.get_yaxis().set_visible(False)
                ax.set_frame_on(False)
                librosa.display.specshow(mel)
                plt.savefig(
                    f"{visible_folder}/img_{'%05d' % (1+i+frame_num)}.jpg", 
                    dpi=1000, 
                    bbox_inches="tight",
                    pad_inches=0
                )
                plt.close('all')

                mel_3c = cv2.imread(f"{visible_folder}/img_{'%05d' % (1+i+frame_num)}.jpg")
                mel_3c = cv2.resize(mel_3c, (frame_size, frame_size))
                mel_3cs.append(mel_3c)
            os.system(f"rm {output}/{file_name}.wav")
        else:
            # If no mdist data, create dummy mel images
            mel_3cs = [np.zeros((frame_size, frame_size, 3)) for _ in range(frame_num)]
            for i, mel in enumerate(mel_3cs):
                cv2.imwrite(f"{visible_folder}/img_{'%05d' % (1+i+frame_num)}.jpg", mel)

        # Optionally save extracted face images
        if save_image:
            for i, face in enumerate(select_faces):
                cv2.imwrite(f"{visible_folder}/img_{'%05d' % (1+i)}.jpg", face)

        # Save as npz
        np.savez_compressed(
            f"{output}/{file_name}", 
            faces=select_faces, 
            mel_3cs=mel_3cs, 
            mdist=mdist, 
            latency=latency, 
            conf=conf
        )

        extraction_queue.task_done()

def work(save_avi, resample_rate, device, max_clip_len, skip_num, frame_size, frame_num, start_time, video_num, output):
    """
    Worker process that:
    - Reads video from queue.
    - Detect faces.
    - Compute latency.
    - Put results into extraction_queue for saving.
    """
    try:
        # Load face detector model
        face_detector = get_model(
            model_name="resnet50_2020-07-20",
            max_size=2048, 
            device=device
        )
        face_detector.eval()

        # Load SyncNet
        __S__ = S(num_layers_in_fc_layers=1024).cuda().half().eval()
        loaded_state = torch.load("syncnet_v2.model", map_location=lambda storage, loc: storage)
        state = __S__.state_dict()
        for name, param in loaded_state.items():
            state[name].copy_(param)
    except Exception as e:
        log(e)
        return

    while True:
        file_path = queue.get()
        if file_path is None:
            queue.task_done()
            return

        if fail_queue.qsize() > 96:
            # If too many failures, skip
            queue.task_done()
            continue

        file_name = file_path.split("/")[-1].split(".")[0]

        if os.path.isfile(f"{output}/{file_name}.npz"):
            log(f"skip {file_name}")
            queue.task_done()
            continue

        try:
            tik = time.time()
            videoclip = VideoFileClip(file_path)
            start, end = 0, min(max_clip_len, videoclip.duration)

            # Extract frames at 25 fps
            frames = []
            for frame in videoclip.subclip(start, end).iter_frames(fps=25):
                frames.append(frame)
            frames = np.array(frames)
            log(f"video reading time: {time.time() - tik}")

            # Detect faces
            all_boxes, all_probs = crop_faces(file_path, face_detector, frames, skip_num)
            all_sizes = [max((box[3]-box[1]), (box[2]-box[0]))/2 for box in all_boxes]
            all_ys = [(box[1]+box[3])/2 for box in all_boxes]
            all_xs = [(box[0]+box[2])/2 for box in all_boxes]

            # Add slight random jitter
            all_ys = [y + np.random.randint(-int(y*0.015), int(y*0.015)+1) for y in all_ys]
            all_xs = [x + np.random.randint(-int(x*0.015), int(x*0.015)+1) for x in all_xs]

            # Smooth bboxes over time
            all_sizes = uniform_filter1d(all_sizes, size=13)
            all_ys = uniform_filter1d(all_ys, size=13)
            all_xs = uniform_filter1d(all_xs, size=13)

            faces, probs = [], []
            for frame, size, y, x, prob in zip(frames, all_sizes, all_ys, all_xs, all_probs):
                face = apply_crop(frame, size, y, x)
                faces.append(cv2.resize(face, (frame_size, frame_size))[:,:,::-1])
                probs.append(prob)
            log(f"video processing time: {time.time() - tik}")
            tik = time.time()
            del frames
            del videoclip
            gc.collect()

            # Process audio
            try:
                audioclip = AudioFileClip(file_path).subclip(start, end)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    audio = audioclip.to_soundarray()[:, 0]
                audio_rate = int(len(audio)/(end - start))
                audio = torch.tensor(librosa.resample(audio, audio_rate, resample_rate)).float()
                mdist, offset, conf = compute_latency(__S__, faces, audio, resample_rate, batch_size=8, vshift=15)
            except Exception as e:
                log("No audio found!", e)
                mdist, offset, conf = [], [], []

            if save_avi:
                # Save as a video file with audio
                videoclip = ImageSequenceClip.ImageSequenceClip(faces, fps=25)
                new_audioclip = CompositeAudioClip([audioclip])
                videoclip.audio = new_audioclip
                videoclip.write_videofile(f"{output}/{file_name}.avi", codec="rawvideo")

            log(f"audio processing time: {time.time() - tik}")
            log(f"offset of {file_name}: {offset}")

            # Uniformly sample frames for saving
            step = len(faces)//frame_num
            indices = [i * step for i in range(frame_num)]
            select_faces = [faces[i] for i in indices]

            extraction_queue.put((file_path, start, end, select_faces, mdist, offset, conf))
        except Exception as e:
            # If processing fails, put the video back in the queue for retry
            queue.put(file_path)
            fail_queue.put((file_path))
            log("=" * 20, file_path)
            log(e)
            log(f"number of failed videos: {fail_queue.qsize()}")
            traceback.print_exc()
            time.sleep(3)
        queue.task_done()

def main(args):
    config = json.load(open("config.json"))
    create_folder(args.output)

    files = [f for f in sorted(os.listdir(args.input)) if f.endswith(".mp4")]
    for f in files:
        queue.put(f"{args.input}/{f}")

    # Start worker processes for face detection & audio sync
    workers = []
    start_time = time.time()
    for _ in range(args.workers):
        worker = Process(target=work, args=(
            args.save_avi,
            config['resample_rate'],
            config['device'],
            config['max_clip_len'],
            config['face_detection_step'],
            config['frame_size'],
            config['frame_num'],
            start_time,
            len(files),
            args.output,
        ))
        worker.start()
        workers.append(worker)

    # Start extractor processes
    extractors = []
    for i in range(args.workers):
        extractor = Process(target=extract, args=(
            args.save_image,
            config['frame_num'],
            config['frame_size'],
            args.output,
        ))
        extractor.start()
        extractors.append(extractor)

    # Wait for all tasks
    queue.join()
    extraction_queue.join()

    # Send termination signals
    for _ in range(args.workers):
        queue.put(None)
    for _ in range(args.workers):
        extraction_queue.put(None)

    for worker in workers:
        worker.join()
    for extractor in extractors:
        extractor.join()

    log(f'average crop time: {(time.time() - start_time) / len(files)}')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--workers",    type=int, required=True, help="Number of parallel worker processes")
    parser.add_argument("--input",      type=str, required=True, help="Input directory of raw videos")
    parser.add_argument("--output",     type=str, required=True, help="Output directory for processed data")
    parser.add_argument("--save-image", action='store_true',     help="Save extracted face images")
    parser.add_argument("--save-avi",   action='store_true',     help="Save processed video with extracted faces")

    args = parser.parse_args()
    main(args)