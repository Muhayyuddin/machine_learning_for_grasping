import sys

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QTableView, QTableWidget, QTableWidgetItem, \
    QAbstractItemView, QComboBox
from PyQt5.QtWidgets import QMessageBox


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.title = "Main window"
        self.left = 100
        self.top = 100
        self.height = 1000
        self.width = 1060
        self.button1 = QPushButton("Compute Grasps",self)
        self.button1.setToolTip(" this button for compute best grasp")
        self.button1.setGeometry(QtCore.QRect(80, 20, 141, 31))
        self.button1.clicked.connect(self.on_click_button1)
        self.button2 = QPushButton("Compute Best Grasp", self)
        self.button2.setToolTip(" put here tooltips")
        self.button2.setGeometry(QtCore.QRect(300, 20, 161, 31))
        self.button2.clicked.connect(self.on_click_button2)
        self.label = QLabel("Computed Grasps",self)
        self.label.setGeometry(QtCore.QRect(60, 70, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.label.setStyleSheet("color : rgb(204, 0, 0)")
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setObjectName("label")
        self.label_2 = QLabel("Best Grasps", self)
        self.label_2.setGeometry(QtCore.QRect(60, 400, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.label_2.setStyleSheet("color : rgb(204, 0, 0)")
        self.label_2.setTextFormat(QtCore.Qt.RichText)
        self.label_2.setObjectName("label_2")
        self.tableView = QTableWidget(self)
        self.tableView.setGeometry(QtCore.QRect(60, 120, 940, 250))
        self.tableView.setObjectName("tableView")
        self.tableView.setColumnCount(9)
        self.tableView.setRowCount(20)
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setGeometry(QtCore.QRect(60, 450, 940, 250))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(9)
        self.tableWidget.setRowCount(20)
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(QtCore.QRect(550, 20, 161, 31))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox_2 = QComboBox(self)
        self.comboBox_2.setGeometry(QtCore.QRect(840, 20, 161, 31))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        _translate = QtCore.QCoreApplication.translate
        self.comboBox.setItemText(0, _translate("comboBox", "pick_clean"))
        self.comboBox.setItemText(1, _translate("comboBox", "pick_serve"))
        self.comboBox.setItemText(2, _translate("comboBox", "pour"))
        self.comboBox.setItemText(3, _translate("comboBox", "push"))
        self.comboBox.setItemText(4, _translate("comboBox", "pull"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "cup"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "pizza cutter"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "glass"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "cup"))
        self.setTableTitle()
        self.setupui()

    def setupui(self):

        self.setWindowTitle(self.title)
        self.setGeometry(self.top,self.left,self.width,self.height)
        self.show()

    def setTableTitle(self):
        self.tableWidget.setItem(0, 0, QTableWidgetItem("Quaternion.w"))
        self.tableWidget.setItem(0, 1, QTableWidgetItem("Quaternion.x"))
        self.tableWidget.setItem(0, 2, QTableWidgetItem("Quaternion.y"))
        self.tableWidget.setItem(0, 3, QTableWidgetItem("Quaternion.z"))
        self.tableWidget.setItem(0, 4, QTableWidgetItem("Translation.x"))
        self.tableWidget.setItem(0, 5, QTableWidgetItem("Translation.y"))
        self.tableWidget.setItem(0, 6, QTableWidgetItem("Translation.z"))
        self.tableWidget.setItem(0, 7, QTableWidgetItem("object_name"))
        self.tableWidget.setItem(0, 8, QTableWidgetItem("actions"))
        self.tableView.setItem(0, 0, QTableWidgetItem("Quaternion.w"))
        self.tableView.setItem(0, 1, QTableWidgetItem("Quaternion.x"))
        self.tableView.setItem(0, 2, QTableWidgetItem("Quaternion.y"))
        self.tableView.setItem(0, 3, QTableWidgetItem("Quaternion.z"))
        self.tableView.setItem(0, 4, QTableWidgetItem("Translation.x"))
        self.tableView.setItem(0, 5, QTableWidgetItem("Translation.y"))
        self.tableView.setItem(0, 6, QTableWidgetItem("Translation.z"))
        self.tableView.setItem(0, 7, QTableWidgetItem("object_name"))
        self.tableView.setItem(0, 8, QTableWidgetItem("actions"))
    @pyqtSlot()
    def on_click_button1(self):
        x = self.comboBox.currentText()
        QMessageBox.about(self, "Title", x)

    def on_click_button2(self):
        QMessageBox.about(self, "Title", "Please Write Button2 clicked code Here")
        self.tableWidget.setItem(1, 0, QTableWidgetItem("Quaternion.w"))
        self.tableWidget.setItem(1, 1, QTableWidgetItem("Quaternion.x"))
        self.tableWidget.setItem(1, 2, QTableWidgetItem("Quaternion.y"))
        self.tableWidget.setItem(1, 3, QTableWidgetItem("Quaternion.z"))
        self.tableWidget.setItem(1, 4, QTableWidgetItem("Translation.x"))
        self.tableWidget.setItem(1, 5, QTableWidgetItem("Translation.y"))
        self.tableWidget.setItem(1, 6, QTableWidgetItem("Translation.z"))
        self.tableWidget.setItem(1, 7, QTableWidgetItem("object_name"))
        self.tableWidget.setItem(1, 8, QTableWidgetItem("actions"))


app = QApplication(sys.argv)
ex = Window()
sys.exit(app.exec_())
