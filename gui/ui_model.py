from gui.ui.ui_window import Ui_MainWindow
from gui.ui_draw import *
from PIL import Image, ImageQt
import random, io, os
import numpy as np
import torch
import torchvision.transforms as transforms
from util import task, util
from dataloader.image_folder import make_dataset
from model import create_model
from util.visualizer import Visualizer
from PyQt5.QtWidgets import QMessageBox, QDesktopWidget
from PyQt5.QtCore import QCoreApplication
from gui.face_process.ffhq_align import FaceData
from gui.camera_dlg import CamDlg
import pathlib
import glob
from PyQt5.QtMultimedia import QCameraInfo


class ui_model(QtWidgets.QMainWindow, Ui_MainWindow):
    shape = 'line'
    CurrentWidth = 1

    def __init__(self, opt):
        super(ui_model, self).__init__()
        self.setupUi(self)
        self.opt = opt
        self.show_result_flag = False
        self.opt.loadSize = [256, 256]
        self.visualizer = Visualizer(opt)
        self.model_name = ['celeba_center', 'paris_center', 'imagenet_center', 'place2_center',
                           'celeba_random', 'paris_random','imagenet_random', 'place2_random']
        self.img_root = './datasets/'
        self.img_files = ['celeba-hq', 'paris', 'imagenet', 'place2']

        # show logo
        self.show_logo()
        self.init_msg()

        # original mask
        self.new_painter()

        # selcet model
        self.comboBox.activated.connect(self.load_model)

        # load image
        self.pushButton.clicked.connect(self.load_image)

        # random image
        self.pushButton_2.clicked.connect(self.random_image)

        # save result
        self.pushButton_4.clicked.connect(self.save_result)

        # draw/erasure the mask
        self.radioButton.toggled.connect(lambda: self.draw_mask('line'))
        self.radioButton_2.toggled.connect(lambda: self.draw_mask('rectangle'))
        self.spinBox.valueChanged.connect(self.change_thickness)
        self.radioButton_2.setChecked(True)
        self.change_thickness(5)
        # erase
        self.pushButton_5.clicked.connect(self.clear_mask)

        # fill image, image process
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.pushButton_3.clicked.connect(self.fill_mask)

        # show the result
        self.pushButton_6.clicked.connect(self.show_result)

        # show manuals and about
        self.btn_manuals.clicked.connect(self.show_manuals)

        # init face model
        self.face_model = FaceData()
        self.cropped_face = None
        self.btn_crop_face.clicked.connect(self.crop_image)

        # init camera window
        self.btn_capture.clicked.connect(self.capture_image)
        self.cap_tmp_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), '.tmpdir')

        # update status bar
        self.statusbar.showMessage("    Welcome", 3000)

        self.setFixedSize(self.size())

    # setting up the actions
    @QtCore.pyqtSlot()
    def on_actionLoad_triggered(self):
        self.load_image()

    @QtCore.pyqtSlot()
    def on_actionManuals_triggered(self):
        self.show_manuals()
    
    @QtCore.pyqtSlot()
    def on_actionAbout_triggered(self):
        self.show_about()

    @QtCore.pyqtSlot()
    def on_actionManuals_triggered(self):
        self.show_manuals()

    @QtCore.pyqtSlot()
    def on_actionSave_triggered(self):
        self.save_result()
    
    @QtCore.pyqtSlot()
    def on_actionDraw_Clear_triggered(self):
        self.clear_mask()
    
    @QtCore.pyqtSlot()
    def on_actionCrop_Face_triggered(self):
        self.crop_image()
    
    @QtCore.pyqtSlot()
    def on_actionRandom_triggered(self):
        self.random_image()

    @QtCore.pyqtSlot()
    def on_actionFill_triggered(self):
        self.fill_mask()

    @QtCore.pyqtSlot()
    def on_actionOriginal_Output_triggered(self):
        self.show_result()
    
    @QtCore.pyqtSlot()
    def on_action_Quit_triggered(self):
        QCoreApplication.quit()

    def init_msg(self):
        self.strmsg_check_load = "Please load an image by clicking 'capture', 'load' or 'random' first.\nKindly refer to 'Manuals' for more details."
        self.strmsg_check_model = "Please choose a model from the 'Options' list first.\nKindly refer to 'Manuals' for more details."

    def showImage(self, fname):
        """Show the masked images"""
        value = self.comboBox.currentIndex()
        if self.cropped_face is not None:
            img = self.cropped_face
        else:
            img = Image.open(fname).convert('RGB')
        self.statusbar.showMessage("    Successfully loaded image.", 2000)
        self.img_original = img.resize(self.opt.loadSize)
        if value > 4:
            self.img = self.img_original
        else:
            self.img = self.img_original.copy()
            sub_img = Image.fromarray(np.uint8(255*np.ones((128, 128, 3))))
            mask = Image.fromarray(np.uint8(255*np.ones((128, 128))))
            self.img.paste(sub_img, box=(64, 64), mask=mask)
        self.show_image = ImageQt.ImageQt(self.img)
        self.new_painter(self.show_image)

    def show_result(self):
        """Show the results and original image"""
        if self.show_result_flag:
            self.show_result_flag = False
            new_pil_image = Image.fromarray(util.tensor2im(self.img_out.detach()))
            new_qt_image = ImageQt.ImageQt(new_pil_image)
        else:
            self.show_result_flag = True
            new_qt_image = ImageQt.ImageQt(self.img_original)
        self.graphicsView_2.scene = QtWidgets.QGraphicsScene()
        item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(new_qt_image))
        self.graphicsView_2.scene.addItem(item)
        self.graphicsView_2.setScene(self.graphicsView_2.scene)

    def show_logo(self):
        """Show the logo of NTU and Monash"""
        img = QtWidgets.QLabel(self)
        img.setGeometry(90, 45, 140, 50)
        # read images
        pixmap = QtGui.QPixmap("./gui/logo/NTU_logo.jpg")
        pixmap = pixmap.scaled(140, 140, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        img.setPixmap(pixmap)
        img.show()
        img1 = QtWidgets.QLabel(self)
        img1.setGeometry(695, 45, 175, 50)
        # read images
        pixmap1 = QtGui.QPixmap("./gui/logo/monash_logo.png")
        pixmap1 = pixmap1.scaled(175, 175, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        img1.setPixmap(pixmap1)
        img1.show()

    def load_model(self):
        """Load different kind models for different datasets and mask types"""
        self.statusbar.showMessage("    Loading model...")
        value = self.comboBox.currentIndex()
        if value == 0:
            QMessageBox.warning(self, "Warning", "Please choose a model, Do NOT choose 'None'.")
            self.statusbar.showMessage("")
            return
        else:
            # define the model type and dataset type
            index = value-1
            self.opt.name = self.model_name[index]
            self.opt.img_file = self.img_root + self.img_files[index % len(self.img_files)]
        if self.opt.name in ['celeba_center', 'celeba_random']:
            self.btn_crop_face.setEnabled(True)
        else:
            self.btn_crop_face.setEnabled(False)
        self.model = create_model(self.opt)
        self.statusbar.showMessage("    Model [%s] is loaded" % self.comboBox.currentText(), 3000)

    def load_image(self):
        """Load the image"""
        if self.init_databuffer():
            return

        self.fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'select the image', self.opt.img_file, 'Image files(*.jpg *.png *.jpeg *.JPG *.PNG *.JPEG)')
        if len(self.fname) == 0:
            QMessageBox.warning(self, "Warning", "Please open an image.")
            return

        self.btn_crop_face.setEnabled(True)
        self.showImage(self.fname)

    def crop_image(self):
        if self.opt.name in ['celeba_center', 'celeba_random']:
            cropped_face = self.face_model.normalize(self.fname)
            if cropped_face is None:
                QMessageBox.warning(self, "Warning", "Please choose an image that contains human faces.")
                return
            self.cropped_face = self.center_crop_image(cropped_face, (246, 246))
            self.showImage(self.fname)
        else:
            QMessageBox.warning(self, "Warning", "This option is only available for CelebA face model.")
            return

    def random_image(self):
        """Random load the test image"""
        if self.init_databuffer():
            return
        self.btn_crop_face.setEnabled(False)

        # read random mask
        if self.opt.mask_file != "none":
            mask_paths, mask_size = make_dataset(self.opt.mask_file)
            item = random.randint(0, mask_size - 1)
            self.mname = mask_paths[item]

        image_paths, image_size = make_dataset(self.opt.img_file)
        item = random.randint(0, image_size-1)
        self.fname = image_paths[item]
        self.showImage(self.fname)

    def save_result(self):
        """Save the results to the disk"""
        if not hasattr(self, 'fname') or len(self.fname) == 0:
            QMessageBox.warning(self, "Warning", self.strmsg_check_load)
            return

        util.mkdir(self.opt.results_dir)
        img_name = self.fname.split('/')[-1]
        data_name = self.opt.img_file.split('/')[-1].split('.')[0]

        # save the original image
        original_name = '%s_%s_%s' % ('original', data_name, img_name)
        original_path = os.path.join(self.opt.results_dir, original_name)
        img_original = util.tensor2im(self.img_truth)
        util.save_image(img_original, original_path)

        # save the mask
        mask_name = '%s_%s_%d_%s' % ('mask', data_name, self.PaintPanel.iteration, img_name)
        mask_path = os.path.join(self.opt.results_dir, mask_name)
        img_mask = util.tensor2im(self.img_m)
        util.save_image(img_mask, mask_path)

        # save the results
        result_name = '%s_%s_%d_%s' % ('result', data_name, self.PaintPanel.iteration, img_name)
        result_path = os.path.join(self.opt.results_dir, result_name)
        img_result = util.tensor2im(self.img_out)
        util.save_image(img_result, result_path)
        print("[INFO] Saved images to %s*_%s_%s" % (self.opt.results_dir, data_name, img_name))
        self.statusbar.showMessage("    Saved images to %s*_%s_%s" % (self.opt.results_dir, data_name, img_name), 5000)

    def new_painter(self, image=None):
        """Build a painter to load and process the image"""
        # painter
        self.PaintPanel = painter(self, image)
        self.PaintPanel.close()
        self.stackedWidget.insertWidget(0, self.PaintPanel)
        self.stackedWidget.setCurrentWidget(self.PaintPanel)

    def change_thickness(self, num):
        """Change the width of the painter"""
        self.CurrentWidth = num
        self.PaintPanel.CurrentWidth = num

    def draw_mask(self, maskStype):
        """Draw the mask"""
        if maskStype == 'rectangle':
            self.change_thickness(5)
        else:
            self.change_thickness(self.spinBox.value())
        self.shape = maskStype
        self.PaintPanel.shape = maskStype

    def clear_mask(self):
        """Clear the mask"""
        if not hasattr(self, 'fname') or len(self.fname) == 0:
            QMessageBox.warning(self, "Warning", self.strmsg_check_load)
            return

        self.showImage(self.fname)
        if self.PaintPanel.Brush:
            self.PaintPanel.Brush = False
        else:
            self.PaintPanel.Brush = True

    def set_input(self):
        """Set the input for the network"""
        # get the test mask from painter
        self.PaintPanel.saveDraw()
        buffer = QtCore.QBuffer()
        buffer.open(QtCore.QBuffer.ReadWrite)
        self.PaintPanel.map.save(buffer, 'PNG')
        pil_im = Image.open(io.BytesIO(buffer.data()))

        # transform the image to the tensor
        img = self.transform(self.img)
        value = self.comboBox.currentIndex()
        if value > 4:
            mask = torch.autograd.Variable(self.transform(pil_im)).unsqueeze(0)
            # mask from the random mask
            # mask = Image.open(self.mname)
            # mask = torch.autograd.Variable(self.transform(mask)).unsqueeze(0)
            mask = (mask < 1).float()
        else:
            mask = task.center_mask(img).unsqueeze(0)
        if len(self.opt.gpu_ids) > 0:
            img = img.unsqueeze(0).cuda(self.opt.gpu_ids[0], True)
            mask = mask.cuda(self.opt.gpu_ids[0], True)

        # get I_m and I_c for image with mask and complement regions for training
        mask = mask
        self.img_truth = img * 2 - 1
        self.img_m = mask * self.img_truth
        self.img_c = (1 - mask) * self.img_truth

        return self.img_m, self.img_c, self.img_truth, mask

    def fill_mask(self):
        """Forward to get the generation results"""
        if not hasattr(self, 'img'):
            QMessageBox.warning(self, "Warning", self.strmsg_check_load)
            return

        img_m, img_c, img_truth, mask = self.set_input()
        if self.PaintPanel.iteration < 100:
            with torch.no_grad():
                # encoder process
                distributions, f = self.model.net_E(img_m)
                q_distribution = torch.distributions.Normal(distributions[-1][0], distributions[-1][1])
                #q_distribution = torch.distributions.Normal( torch.zeros_like(distributions[-1][0]), torch.ones_like(distributions[-1][1]))
                z = q_distribution.sample()

                # decoder process
                scale_mask = task.scale_pyramid(mask, 4)
                self.img_g, self.atten = self.model.net_G(z, f_m=f[-1], f_e=f[2], mask=scale_mask[0].chunk(3, dim=1)[0])
                self.img_out = (1 - mask) * self.img_g[-1].detach() + mask * img_m

                score = self.model.net_D(self.img_out).mean()
                # get score
                self.label_6.setText(str(round(score.item(),3)))
                self.PaintPanel.iteration += 1

        self.show_result_flag = True
        self.show_result()

    def show_manuals(self):
        msg = "<ol> \
            <li>Select a model from 'Options'</li> \
            <li>Click the 'capture', 'load' or 'random' button to get an input image.</li> \
            <li><em>[Optional]</em> For face image and model, click the 'crop face' button to automatically detect and align face</li> \
            <li>If you choose a random model, click the 'draw/clear' button to input free_form mask.</li> \
            <li>If you choose a center model, the center mask has been given.</li> \
            <li>click 'fill' button to get multiple results.</li> \
            <li>click 'save' button to save the results.</li> \
            </ol>"
        QMessageBox.information(self, "Manuals", msg)

    def show_about(self):
        msg = "<H2>Pluralistic Image Completion</H2>\
            <p>This demo is based on the papers <em>Chuanxia Zheng, Tat-Jen Cham, Jianfei Cai, Pluralistic image completion, <b>CVPR</b>, 2019</em>\
                and <em>Chuanxia Zheng, Tat-Jen Cham, Jianfei Cai, Pluralistic free-form image completion, <b>IJCV</b>, 2021.</em></p>\
            <br>\
            <p><b>Maintainers</b>: <a href='https://chuanxiaz.com'>Chuanxia Zheng</a>, <a href='https://donydchen.github.io'>Yuedong Chen</a></p>\
            <p><b>Source</b>: <a href='https://github.com/lyndonzheng/Pluralistic-Inpainting'>https://github.com/lyndonzheng/Pluralistic-Inpainting</a></p>\
            <p><b>License</b>: <a href='http://creativecommons.org/licenses/by-nc/4.0/'>Creative Commons Attribution-NonCommercial 4.0 International License</a></p>\
            <p><b>Acknowledgements</b>: The demo is initially created by Chuanxia when he was a PhD student at NTU, funded by <a href='https://dr.ntu.edu.sg/handle/10356/79124'>BeingThere Centre@IMI.NTU</a>.</p>"
        QMessageBox.about(self, "About", msg)

    def center_window(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def capture_image(self):
        if self.init_databuffer():
            return

        if len(QCameraInfo.availableCameras()) < 1:
            QMessageBox.warning(self, "Warning", "Sorry, your system does not have a camera.")
            return

        self.camera_capture = CamDlg(self.cap_tmp_dir)
        self.camera_capture.cap_clicked.connect(self.on_cam_dlg_confirm)
        self.camera_capture.show()
    
    def on_cam_dlg_confirm(self, text):
        print("[INFO] Temporarily save photo to %s" % text)
        self.fname = text
        while True:
            if os.path.isfile(text):
                # center crop the image
                im = Image.open(text).convert('RGB')
                im = self.center_crop_image(im)
                im.save(text)
                break
        self.btn_crop_face.setEnabled(True)
        self.showImage(self.fname)

    def clear_tmpdir(self):
        is_clear = False
        for filepath in glob.glob(os.path.join(self.cap_tmp_dir, '*')):
            os.remove(filepath)
            is_clear = True
        if is_clear:
            print("[INFO] Removed all cached data.")

    def init_databuffer(self):
        if self.opt.name not in self.model_name:
            QMessageBox.warning(self, "Warning", self.strmsg_check_model)
            return 1
        self.cropped_face = None
        self.clear_tmpdir()
        return 0

    def center_crop_image(self, im, new_size=None):
        width, height = im.size
        if new_size is None:
            new_width = min(width, height)
            new_height = min(width, height)
        else:
            new_width, new_height = new_size
        left = (width - new_width)/2
        top = (height - new_height)/2
        right = (width + new_width)/2
        bottom = (height + new_height)/2
        im = im.crop((left, top, right, bottom))
        return im
