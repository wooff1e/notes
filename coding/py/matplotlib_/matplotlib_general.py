import numpy as np
import matplotlib.pyplot as plt
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.rc.html

plt.rcParams['figure.figsize'] = 8, 4
plt.rcParams['figure.facecolor'] = 'ffffff'

font = {'family' : 'monospace',
        'weight' : 'bold',
        'size'   : 'larger'} # or       'size'   : 22}

plt.rc('font', **font)  # pass in the font dict as kwargs

# 100 values evenly spaced in [0, 10] 
x = np.linspace(0, 10, 100) 
y = 4 + 2 * np.sin(2 * x)

#1) pyplot - simple, pandas uses mostly this
plt.plot([10,20,30])

#2) for common layouts of subplots, including the enclosing figure object, in a single call.
fig, ax = plt.subplots()
ax.plot([10,20,30])

#3) matplotlib OOP - flexible
fig = plt.figure()      # take an empty "page"
ax = fig.subplots()     # allocate space for 1 chart
ax.plot(...)            # place objects - text, image, lines etc.

help(fig)

# overlay to images
import cv2
def read_img(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return img/255

gt = read_img('gt.png')
output = read_img('out.png')

plt.imshow(gt)
plt.imshow(output, cmap='gray', alpha=0.4)
 


plt.subplots(2, 2, constrained_layout=True)
plt.subplot(221)  # (rows, cols, index)
plt.imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('left')

plt.subplot(222)
plt.imshow(cv2.cvtColor(left_norm, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('left_norm')

plt.subplot(223) 
plt.imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('right')

plt.subplot(224)
plt.imshow(cv2.cvtColor(right_norm, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('right_norm')

plt.show()

