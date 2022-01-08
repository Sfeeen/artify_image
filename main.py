import glob
import os
import sys
from os.path import expanduser
from shutil import copyfile, rmtree
import glob
import pickle
from math import floor
from PyQt5.QtGui import QIcon, QWheelEvent, QColor, QFont, QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QTableWidgetItem, \
    QHeaderView, QWidget, QCheckBox, QHBoxLayout, QComboBox, QColorDialog, QPushButton, QDialog, QDialogButtonBox, \
    QFileDialog
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
from random import randrange
import cv2

import layout

GENERAL_IMAGE_WIDTH = 281
GENERAL_IMAGE_HEIGHT = 500

exe_path = os.path.dirname(sys.argv[0])
convert_images_path = f"{exe_path}/images_converted"
temp_path = f"{exe_path}/temp"
PICKLE_NAME = convert_images_path + "/color_list.pk"
PICKLE_NAME_THREAD = convert_images_path + "/convertthread.pk"
PICKLE_NAME_ARTIFY_THREAD = convert_images_path + "/artifythread.pk"


# pyuic5 -x layout.ui -o layout.py
# pyinstaller.exe --onefile --windowed --icon="assets/artify.ico" main.py --add-data "assets/;assets"


def resize_image_to_standard_size(im):
    width, height = im.size

    if (GENERAL_IMAGE_WIDTH > GENERAL_IMAGE_HEIGHT and height > width) or (
            GENERAL_IMAGE_WIDTH < GENERAL_IMAGE_HEIGHT and height < width):
        im = im.rotate(90, expand=True)
        width, height = im.size

    if width < GENERAL_IMAGE_WIDTH or height < GENERAL_IMAGE_HEIGHT:
        print("Error image size to small")
        return

    width_factor = GENERAL_IMAGE_WIDTH / width
    height_factor = GENERAL_IMAGE_HEIGHT / height

    if (width_factor > height_factor):
        target_height = int(width_factor * height)
        target_width = int(width_factor * width)
        im = im.resize((target_width, target_height))
    else:
        target_height = int(height_factor * height)
        target_width = int(height_factor * width)
        im = im.resize((target_width, target_height))

    im = im.crop((0, 0, GENERAL_IMAGE_WIDTH, GENERAL_IMAGE_HEIGHT))

    return im


def get_average_color(im, use_chrominance=False):
    width, height = im.size
    # if width != GENERAL_IMAGE_WIDTH or height != GENERAL_IMAGE_HEIGHT:
    #     print("Not a standard size image!")
    #     return

    if use_chrominance:
        sum_y = 0
        sum_u = 0
        sum_v = 0
        yuv_img = im.convert('YCbCr')
        for x in range(width):
            for y in range(height):
                y, u, v = yuv_img.getpixel((x, y))
                sum_y += y
                sum_u += u
                sum_v += v

        avg_y = int(sum_y / (height * width))
        avg_u = int(sum_u / (height * width))
        avg_v = int(sum_v / (height * width))

        avg_r, avg_g, avg_b = YUV2RGB(avg_y, avg_u, avg_v)
        return (avg_r, avg_g, avg_b)

    else:
        sum_r = 0
        sum_g = 0
        sum_b = 0
        rgb_im = im.convert('RGB')
        for x in range(width):
            for y in range(height):
                r, g, b = rgb_im.getpixel((x, y))
                sum_r += r
                sum_b += b
                sum_g += g

        avg_r = int(sum_r / (height * width))
        avg_g = int(sum_g / (height * width))
        avg_b = int(sum_b / (height * width))

        return (avg_r, avg_g, avg_b)


def RGB2YUV(R, G, B):
    (R, G, B) = input
    Y = int(0.299 * R + 0.587 * G + 0.114 * B)
    U = int(-0.147 * R + -0.289 * G + 0.436 * B)
    V = int(0.615 * R + -0.515 * G + -0.100 * B)

    return (Y, U, V)


def YUV2RGB(Y, U, V):
    R = int(Y + 1.14 * V)
    G = int(Y - 0.39 * U - 0.58 * V)
    B = int(Y + 2.03 * U)

    return (R, G, B)


def show_image_next_to_average_color(im):
    total_img = Image.new('RGB', (GENERAL_IMAGE_WIDTH * 3, GENERAL_IMAGE_HEIGHT), (0, 0, 0))
    total_img.paste(im, (0, 0))

    avg_r, avg_g, avg_b = get_average_color(im, use_chrominance=False)
    img_avg = Image.new('RGB', (GENERAL_IMAGE_WIDTH, GENERAL_IMAGE_HEIGHT), (avg_r, avg_g, avg_b))
    total_img.paste(img_avg, (GENERAL_IMAGE_WIDTH, 0))

    images_per_row_and_column = 20

    small_img = im.resize(
        (int(GENERAL_IMAGE_WIDTH / images_per_row_and_column), int(GENERAL_IMAGE_HEIGHT / images_per_row_and_column)))
    for x in range(images_per_row_and_column):
        for y in range(images_per_row_and_column):
            total_img.paste(small_img, (
                2 * GENERAL_IMAGE_WIDTH + x * int(GENERAL_IMAGE_WIDTH / images_per_row_and_column),
                y * int(GENERAL_IMAGE_HEIGHT / images_per_row_and_column)))

    total_img.show()


def store_color_list(l):
    with open(PICKLE_NAME, 'wb') as handle:
        pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)


def retrieve_color_list():
    with open(PICKLE_NAME, 'rb') as handle:
        l = pickle.load(handle)

    return l

def store_threaddata(l):
    with open(PICKLE_NAME_THREAD, 'wb') as handle:
        pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)


def retrieve_threaddata():
    with open(PICKLE_NAME_THREAD, 'rb') as handle:
        l = pickle.load(handle)

    return l

def store_artifythreaddata(l):
    with open(PICKLE_NAME_ARTIFY_THREAD, 'wb') as handle:
        pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)


def retrieve_artifythreaddata():
    with open(PICKLE_NAME_ARTIFY_THREAD, 'rb') as handle:
        l = pickle.load(handle)
    return l


def convert_and_index_all_images(images):
    if not os.path.exists(convert_images_path):
        os.makedirs(convert_images_path)
    else:
        rmtree(convert_images_path)
        os.makedirs(convert_images_path)

    color_list = []

    for index, image in enumerate(images):
        print(image)
        im = Image.open(image)
        im = resize_image_to_standard_size(im)
        avg_r, avg_g, avg_b = get_average_color(im, use_chrominance=False)

        color_list.append((avg_r, avg_g, avg_b))
        im.save(f"{convert_images_path}/{index}.jpg")

    store_color_list(color_list)


def color_distance(r, g, b, r2, g2, b2):
    dist = abs(r2 - r) + abs(g2 - g) + abs(b2 - b)
    return dist


def get_closest_image(r, g, b, color_list, used_indexes, do_not_reuse=False):
    smallest_distance = 99999
    best_index = -1

    for index, color in enumerate(color_list):
        if index in used_indexes and do_not_reuse:
            continue
        r2, g2, b2 = color
        if (color_distance(r, g, b, r2, g2, b2) < smallest_distance):
            smallest_distance = color_distance(r, g, b, r2, g2, b2)
            best_index = index

    return best_index


IMAGES_PER_ROW = 60


def artify_image(path):
    color_list = retrieve_color_list()

    main_image = Image.open(path)
    width, height = main_image.size
    small_img_width = int(width / IMAGES_PER_ROW)
    small_img_heigth = int(small_img_width * GENERAL_IMAGE_HEIGHT / GENERAL_IMAGE_WIDTH)
    images_vertical = floor(height / small_img_heigth)

    used_indexes = []

    for x in range(IMAGES_PER_ROW):
        for y in range(images_vertical):
            area = (0 + x * small_img_width, 0 + y * small_img_heigth, 0 + (x + 1) * small_img_width,
                    0 + (y + 1) * small_img_heigth)
            cropped_img = main_image.crop(area)
            r, g, b = get_average_color(cropped_img)
            index = get_closest_image(r, g, b, color_list, used_indexes, True)
            if index < 0:
                print("Not enough images to fill the image!")
                return
            used_indexes.append(index)
            closest_im = Image.open(f"{convert_images_path}/{index}.jpg")
            closest_im_resized = closest_im.resize((small_img_width, small_img_heigth))

            main_image.paste(closest_im_resized, (0 + x * small_img_width, 0 + y * small_img_heigth))

    main_image.show()


def get_images(folname, recursively):
    imglist = []

    if recursively:
        folders = next(os.walk(folname))[1]
        for folder in folders:
            imglist.extend(get_images(os.path.join(folname, folder), True))

        # put a slash in dir name if needed
        if folname[-1] != chr(92):
            folname = folname + chr(92)

        # iterate the files in dir using glob
        for path in glob.glob(folname + '*.*'):

            filename = path.split("\\")[-1]
            extension = ""
            if '.' in filename:
                extension = filename.lower().split('.')[-1]

            if extension == "jpg":
                imglist.append(path)
    else:
        for path in glob.glob(folname + '*.jpg'):
            imglist.append(path)

    return imglist

class ConvertThread(QThread):

    conversion_done = pyqtSignal()
    progress_conversion = pyqtSignal(object)
    images_found = pyqtSignal(object)

    def __init__(self):
        QThread.__init__(self)

    def run(self):
        thread_data = retrieve_threaddata()
        images = get_images(thread_data["path"], thread_data["recursively"])
        self.images_found.emit(len(images))


        percent_per_image = 100 / len(images)
        percent = 0

        if not os.path.exists(convert_images_path):
            os.makedirs(convert_images_path)
        else:
            rmtree(convert_images_path)
            os.makedirs(convert_images_path)

        color_list = []

        for index, image in enumerate(images):
            print(image)
            im = Image.open(image)
            im = resize_image_to_standard_size(im)
            avg_r, avg_g, avg_b = get_average_color(im, use_chrominance=False)

            color_list.append((avg_r, avg_g, avg_b))
            im.save(f"{convert_images_path}/{index}.jpg")

            percent += percent_per_image
            self.progress_conversion.emit(percent)

        store_color_list(color_list)
        self.conversion_done.emit()

class ArtifyThread(QThread):

    artifying_done = pyqtSignal()
    update_image = pyqtSignal(object)

    def __init__(self):
        QThread.__init__(self)

    def run(self):
        artify_thread_data = retrieve_artifythreaddata()

        IMAGES_PER_ROW = artify_thread_data["img_per_row"]
        GENERAL_IMAGE_HEIGHT = artify_thread_data["g_img_heigth"]
        GENERAL_IMAGE_WIDTH = artify_thread_data["g_img_width"]
        do_not_reuse_images = artify_thread_data["do_not_reuse"]

        color_list = retrieve_color_list()
        imgpath = temp_path + "/random.jpg"
        main_image = Image.open(imgpath)
        width, height = main_image.size
        small_img_width = int(width / IMAGES_PER_ROW)
        small_img_heigth = int(small_img_width * GENERAL_IMAGE_HEIGHT / GENERAL_IMAGE_WIDTH)
        images_vertical = floor(height / small_img_heigth)

        used_indexes = []

        percent_per_image = 100 / (IMAGES_PER_ROW * images_vertical)
        percent = 0

        for x in range(IMAGES_PER_ROW):
            for y in range(images_vertical):
                area = (0 + x * small_img_width, 0 + y * small_img_heigth, 0 + (x + 1) * small_img_width,
                        0 + (y + 1) * small_img_heigth)
                cropped_img = main_image.crop(area)
                r, g, b = get_average_color(cropped_img)
                index = get_closest_image(r, g, b, color_list, used_indexes, do_not_reuse_images)
                if index < 0:
                    print("Not enough images to fill the image!")
                    return
                used_indexes.append(index)
                closest_im = Image.open(f"{convert_images_path}/{index}.jpg")
                closest_im_resized = closest_im.resize((small_img_width, small_img_heigth))

                main_image.paste(closest_im_resized, (0 + x * small_img_width, 0 + y * small_img_heigth))

                imgpath = temp_path + "/random.jpg"
                main_image.save(imgpath)

                percent += percent_per_image
                self.update_image.emit(percent)

        self.artifying_done.emit()



class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.ui = layout.Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon(resource_path("assets/artify.ico")))

        self.setWindowTitle("Image artify")
        # self.showMaximized()

        self.image_db_path = ""

        self.ui.progress_bar.setValue(0)

        self.ui.set_image_database_path_btn.clicked.connect(self.set_database_path)
        self.ui.convert_and_index_btn.clicked.connect(self.convert_images)
        self.ui.show_random_converted_image_btn.clicked.connect(self.show_random_converted_image)
        self.ui.load_main_image_btn.clicked.connect(self.load_main_image)
        self.ui.save_image_btn.clicked.connect(self.save_image)
        self.ui.artify_main_image_btn.clicked.connect(self.artify_main_image)

        self.threads = [] #for an unknown reason this is required
        convt = ConvertThread()
        convt.conversion_done.connect(self.conversion_done)
        convt.images_found.connect(self.conversion_images_found)
        convt.progress_conversion.connect(self.progress_conversion)
        self.threads.append(convt) #for an unknown reason this is required
        artift = ArtifyThread()
        artift.artifying_done.connect(self.artifying_done)
        artift.update_image.connect(self.update_artify_image)
        self.threads.append(artift) #for an unknown reason this is required

    def conversion_images_found(self, amount):
        self.print_feedback(f"Found {amount} images!")
        self.ui.progress_bar.setValue(0)

    def progress_conversion(self, percent):
        self.ui.progress_bar.setValue(int(percent))

    def conversion_done(self):
        self.print_feedback(f"Conversion done")

    def set_database_path(self):
        my_dir = QFileDialog.getExistingDirectory(
            self,
            "Open a folder",
            expanduser("~"),
            QFileDialog.ShowDirsOnly
        )

        self.clear_feedback()
        self.image_db_path = my_dir + "/"
        self.ui.image_db_label.setText(str(self.image_db_path))
        self.print_feedback(f"Database folder set to: {my_dir}")

    def convert_images(self):
        self.clear_feedback()
        if len(self.image_db_path) < 1:
            self.print_feedback(f"No folder path selected!")
            return

        self.print_feedback(f"Counting images...")

        thread_data = {}
        thread_data["recursively"] = self.ui.search_folder_recursively_checkbox.isChecked()
        thread_data["path"] = self.image_db_path
        store_threaddata(thread_data)
        self.threads[0].start()

    def update_artify_image(self, percent):
        self.ui.progress_bar.setValue(int(percent))

        imgpath = temp_path + "/random.jpg"
        self.show_on_screen(imgpath)


    def artify_main_image(self):
        artify_thread_data = {}
        artify_thread_data["img_per_row"] = int(self.ui.images_per_row.text())
        artify_thread_data["g_img_width"] = int(self.ui.sub_image_width_input.text())
        artify_thread_data["g_img_heigth"] = int(self.ui.sub_image_heigt_input.text())
        artify_thread_data["do_not_reuse"] = self.ui.do_not_reuse_subimages_checkbox.isChecked()
        store_artifythreaddata(artify_thread_data)
        self.clear_feedback()
        self.ui.progress_bar.setValue(0)
        self.threads[1].start()

    def artifying_done(self):
        self.ui.progress_bar.setValue(100)
        self.print_feedback(f"Done artifying")


    def show_random_converted_image(self):
        images = glob.glob(convert_images_path + "/" + '*.jpg')
        index = randrange(len(images))

        im = Image.open(images[index])

        self.GENERAL_IMAGE_WIDTH = int(self.ui.sub_image_width_input.text())
        self.GENERAL_IMAGE_HEIGHT = int(self.ui.sub_image_heigt_input.text())

        total_img = Image.new('RGB', (self.GENERAL_IMAGE_WIDTH * 3, self.GENERAL_IMAGE_HEIGHT), (0, 0, 0))
        total_img.paste(im, (0, 0))

        avg_r, avg_g, avg_b = get_average_color(im, use_chrominance=False)
        img_avg = Image.new('RGB', (self.GENERAL_IMAGE_WIDTH, self.GENERAL_IMAGE_HEIGHT), (avg_r, avg_g, avg_b))
        total_img.paste(img_avg, (self.GENERAL_IMAGE_WIDTH, 0))

        images_per_row_and_column = 20

        small_img = im.resize((int(self.GENERAL_IMAGE_WIDTH / images_per_row_and_column),
                               int(self.GENERAL_IMAGE_HEIGHT / images_per_row_and_column)))
        for x in range(images_per_row_and_column):
            for y in range(images_per_row_and_column):
                total_img.paste(small_img, (
                    2 * self.GENERAL_IMAGE_WIDTH + x * int(self.GENERAL_IMAGE_WIDTH / images_per_row_and_column),
                    y * int(self.GENERAL_IMAGE_HEIGHT / images_per_row_and_column)))

        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

        imgpath = temp_path + "/random.jpg"
        total_img = self.resize_image_to_showscreen(total_img)
        total_img.save(imgpath)
        self.show_on_screen(imgpath)

    def load_main_image(self):
        self.clear_feedback()
        startpath = os.getcwd()
        fname = QFileDialog.getOpenFileName(self, 'Open an image', startpath, "Image files (*.jpg)")
        if not ((".jpg" in fname[0].lower())):
            self.print_feedback(f"No correct extension")
            return

        im = Image.open(fname[0])
        im = self.resize_image_to_showscreen(im)
        imgpath = temp_path + "/random.jpg"
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        im.save(imgpath)
        self.show_on_screen(imgpath)

    def save_image(self):
        path = QFileDialog.getSaveFileName(self, 'Save current view', 'currentview.jpg',
                                           "jpeg (*.jpg);;All Files (*)")[0]
        if not path:
            return

        self.clear_feedback()
        copyfile(temp_path + "/random.jpg", path)
        self.print_feedback(f"Image saved: {path}")

    def resize_image_to_showscreen(self, im):
        self.showheight = 621
        self.showwidth = 1161

        width, height = im.size

        width_factor = self.showwidth / width
        height_factor = self.showheight / height

        smallest_factor = width_factor
        if height_factor < width_factor:
            smallest_factor = height_factor

        im = im.resize((int(smallest_factor * width), int(smallest_factor * height)))
        return im

    def show_on_screen(self, imgpath):
        self.ui.img_label.setPixmap(QPixmap(imgpath))

    def clear_feedback(self):
        self.ui.feedback.setPlainText("")

    def print_feedback(self, str):
        oldtext = self.ui.feedback.toPlainText()
        newtext = oldtext + "\r\n" + str
        self.ui.feedback.setPlainText(newtext)


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec_()


# #
def download_images(folname):
    print(folname)
    folders = next(os.walk(folname))[1]
    for folder in folders:
        download_images(os.path.join(folname, folder))

    # put a slash in dir name if needed
    if folname[-1] != chr(92):
        folname = folname + chr(92)

    # iterate the files in dir using glob
    for path in glob.glob(folname + '*.*'):

        filename = path.split("\\")[-1]
        extension = ""
        if '.' in filename:
            extension = filename.lower().split('.')[-1]

        if extension == "jpg":
            copyfile(path, r"C:\Users\Sven Onderbeke\PycharmProjects\image_art\images/" + filename)
#
#
# download_images("Z:\RepairData\RP\RP073")
