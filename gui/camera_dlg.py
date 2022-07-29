from PyQt5.QtWidgets import QDialog, QErrorMessage
from PyQt5.QtMultimediaWidgets import QCameraViewfinder
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture, QCameraInfo
from PyQt5 import QtCore
from gui.ui.ui_camera import Ui_Dialog
import time
import os


class CamDlg(QDialog):
    '''
    Ref: https://www.geeksforgeeks.org/creating-a-camera-application-using-pyqt5/
    '''
    cap_clicked = QtCore.pyqtSignal(str)

    def __init__(self, tmp_dir):
        super(CamDlg, self).__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)

        # add camera viewer
        self.viewfinder = QCameraViewfinder()
        self.viewfinder.show()
        self.ui.horizontalLayout.addWidget(self.viewfinder)

        self.available_cameras = QCameraInfo.availableCameras()
        self.select_camera(0)

        self.ui.btn_cap.clicked.connect(self.capture_image)
        self.tmp_dir = tmp_dir
        if not os.path.isdir(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        # update camera selector
        self.ui.cam_selector.setToolTip("Select Camera")
        self.ui.cam_selector.setToolTipDuration(2500)
        self.ui.cam_selector.addItems([camera.description() for camera in self.available_cameras])
        self.ui.cam_selector.currentIndexChanged.connect(self.select_camera)

    def select_camera(self, i):
        # init camera
        self.camera = QCamera(self.available_cameras[i])
        self.camera.setViewfinder(self.viewfinder)
        self.camera.setCaptureMode(QCamera.CaptureStillImage)
        self.camera.error.connect(lambda: self.alert(self.camera.errorString()))
        self.current_camera_name = self.available_cameras[i].description()

        # start the camera
        self.camera.start()

        # creating a QCameraImageCapture object
        self.capture = QCameraImageCapture(self.camera)
        self.capture.error.connect(lambda error_msg, error, msg: self.alert(msg))

    def alert(self, msg):
        error = QErrorMessage(self)
        error.showMessage(msg)

    def capture_image(self):
        # time stamp
        timestamp = time.strftime("%d-%b-%Y-%H_%M_%S")
        tmp_path = os.path.join(self.tmp_dir, "%s.jpg" % timestamp)

        # capture the image and save it on the save path
        self.capture.capture(tmp_path)
        
        self.cap_clicked.emit(str(tmp_path))
        self.camera.stop()
        self.close()

    def reject(self) -> None:
        self.camera.stop()
        self.close()
        return super().reject()
