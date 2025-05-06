import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image


def process_frame(frame, w):
    #frame = frame[30:, w:w*2, :]
    frame = frame[30:, 0:w, :]
    return frame


def process_video(video_path, result_path):
    video_capture = cv2.VideoCapture(video_path)

    if video_capture.isOpened():
        w  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print('frames: ', num_frame)
        print('w,h: ', w, h)
        w = w//5
        h -=30

        print('w,h: ', w, h)

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # shortcut to write cv.VideoWriter_fourcc('M','J','P','G')
    result_fps = 30
    video_writer = cv2.VideoWriter(result_path, fourcc, result_fps, (w, h))    


    for i in tqdm(range(201)):
        _, frame = video_capture.read()

        frame = process_frame(frame, w)

        # write
        out_dir = 'data/frames_input'
        name = f'{i:03d}.png'
        cv2.imwrite(f'{out_dir}/{name}', frame)
        video_writer.write(frame)

    video_capture.release()
    video_writer.release()



def process_folder(frame_folder, result_path):
    frame_paths = frame_folder.glob('*.jpg')
    frame_paths = sorted(frame_paths)
    img = cv2.imread(str(frame_paths[0]))
    h, w, c = img.shape

    # video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')    # shortcut to write cv.VideoWriter_fourcc('M','J','P','G')
    result_fps = 24
    video_writer = cv2.VideoWriter(result_path, fourcc, result_fps, (w, h))    


    for path in tqdm(frame_paths):
        frame = cv2.imread(str(path))
        video_writer.write(frame)

    video_writer.release()




if __name__ == '__main__':

    video_path = 'data/video.mp4'
    result_path = 'data/result.mp4'

    #process_video(video_path, result_path)

    frame_folder = Path('data/0060')
    process_folder(frame_folder, result_path)
