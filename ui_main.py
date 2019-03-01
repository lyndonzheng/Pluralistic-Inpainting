import sys
from options.test_options import TestOptions
from gui.ui_model import ui_model
from PyQt5 import QtWidgets


if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    opt = TestOptions().parse()
    my_gui = ui_model(opt)
    my_gui.show()
    sys.exit(app.exec_())
