# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets, QtGui
import sys
import cgitb
import numpy as np

from image import Ui_Form
from PyQt5.QtWidgets import QFileDialog
from chinese_character_recognition_bn import build_graph, inference


class mywindow(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)

    # 定义槽函数
    def openImage(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        self.imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "",
                                                            "All Files (*);; *.jpg;;*.png;;*.jpeg;;*.bmp")
        print(self.imgName)

        # 利用qlabel显示图片
        png = QtGui.QPixmap(self.imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(png)

    def predict(self):
        final_predict_val, final_predict_index, label = inference(self.imgName)
        print('the result : predict index {} predict_val {} character {}'.format(final_predict_index,final_predict_val,
                                                                                 label))


if __name__ == '__main__':
    cgitb.enable(format='text')
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.show()
    sys.exit(app.exec_())
