import sys
from options.test_options import TestOptions
from gui.ui_model import ui_model
from PyQt5 import QtWidgets


__version__ = "0.2.0"


if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    opt = TestOptions().parse()
    my_gui = ui_model(opt)
    my_gui.center_window()
    my_gui.show()
    ret = app.exec_()
    my_gui.clear_tmpdir()
    sys.exit(ret)
