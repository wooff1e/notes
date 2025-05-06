import cv2
import os
import matplotlib
matplotlib.use('agg')


class Focus():
    def __init__(self, image_path):
        self.img_name = os.path.basename(image_path).split('.')[0]
        self.img = cv2.imread(os.path.join('data/bokeh_src', image_path))

    def create_focus_file(self, x, y):
        file_path = "data/focus/" + self.img_name + ".txt"
        print(file_path)
        f = open(file_path, "w")
        text = "Snapshot metadata:\nrequestId:\nptFocus(x,y):"
        text+= str(x) + ' ' + str(y)
        f.write(text)
        f.close()       

    def focus_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.create_focus_file(x, y)

    # Creates a window to display the original image
    # The callback function is attached to this window
    def view_image(self):
        cv2.namedWindow("Image", flags = (cv2.WINDOW_NORMAL))
        cv2.resizeWindow('image', 600,600)
        cv2.setMouseCallback("Image", self.focus_callback)
        cv2.imshow("Image", self.img)

        while(cv2.waitKey() != 27):
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    image_paths = os.listdir(path='data/bokeh_src')
    for image_path in image_paths:
        focus = Focus(image_path)
        focus.view_image()
