import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt



def save_histogram(img, path):
    for i, color in enumerate(['r', 'g', 'b']):
        hist = cv2.calcHist([img], [i], None, [256], [0,256])
        plt.plot(hist, color=color)
        plt.xlim([0,256])

    plt.savefig(path, transparent=False)
    plt.close('all')


def make_gif(frame_folder):
    target_size = (750, 1000)
    frame_paths = glob.glob(f"{frame_folder}/*.jpg")
    frame_paths = sorted(frame_paths)

    frames = []
    for path in frame_paths:
        img = Image.open(path)
        #img = img.resize(target_size, resample=Image.LANCZOS) 
        frames.append(img)


    frame_one = frames[0]
    frame_one.save("data/prior.gif", format="GIF", append_images=frames,
               save_all=True, duration=10, loop=0)

make_gif("data/0060")


def equalize_rgb(image):
    channels = cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['B', 'G', 'R']):
        eq_channels.append(cv2.equalizeHist(ch))

    eq_image = cv2.merge(eq_channels)
    return eq_image