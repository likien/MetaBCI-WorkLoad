import sys
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QRadioButton, \
    QButtonGroup, QLineEdit, QMessageBox, QHBoxLayout, QDesktopWidget, QGridLayout, QTextEdit, QCheckBox
import yaml

# 这里是你的实验代码，你可以将其保存在一个单独的模块中
from paradigm_experiment import run_n_back, run_ssavep, run_arithmetic_task
from functools import partial

TYPE_SSAVEP = "SSaVEP"
TYPE_N_BACK = "N-Back"
TYPE_Arithmetic_Task = "Arithmetic Task"
SOFTWARE_TITLE = 'MetaBCI 实验软件'


def set_window_to_center(win):
    """将窗口居中显示在屏幕上"""
    qr = win.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    win.move(qr.topLeft())


def set_icon(win):
    # 创建一个透明的图标 或者自定义
    transparent_pixmap = QPixmap(32, 32)
    transparent_pixmap.fill(QtCore.Qt.transparent)
    transparent_icon = QIcon(transparent_pixmap)
    win.setWindowIcon(transparent_icon)


class ExperimentLauncher(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(SOFTWARE_TITLE)
        self.setGeometry(100, 100, 300, 200)
        set_window_to_center(self)
        set_icon(self)

        self.init_ui()

    def init_ui(self):
        """初始化主界面UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()

        self.start_button = QPushButton('启动范式实验', self)
        self.start_button.clicked.connect(self.open_experiment_selection)

        layout.addWidget(self.start_button)
        central_widget.setLayout(layout)

    def open_experiment_selection(self):
        """打开实验选择窗口"""
        self.selection_window = ExperimentSelectionWindow()
        self.selection_window.show()


class ExperimentSelectionWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(SOFTWARE_TITLE)
        self.setGeometry(150, 150, 400, 300)
        set_window_to_center(self)
        set_icon(self)

        self.init_ui()

    def init_ui(self):
        """初始化实验选择窗口UI"""
        # 主布局
        main_layout = QVBoxLayout()

        # 创建一个水平布局来放置标签和单选按钮的垂直布局
        horizontal_layout = QHBoxLayout()

        self.label = QLabel('请选择实验：', self)
        horizontal_layout.addWidget(self.label)

        # 创建一个垂直布局来放置单选按钮
        vertical_radio_layout = QVBoxLayout()

        self.radio_button_group = QButtonGroup(self)

        # 添加单选按钮以选择实验
        self.n_back_radio = QRadioButton(TYPE_N_BACK, self)
        self.arithmetic_task_radio = QRadioButton(TYPE_Arithmetic_Task, self)
        self.ssavep_radio = QRadioButton(TYPE_SSAVEP, self)

        # 设置第一个单选按钮为默认选项
        self.n_back_radio.setChecked(True)

        self.radio_button_group.addButton(self.n_back_radio)
        self.radio_button_group.addButton(self.arithmetic_task_radio)
        self.radio_button_group.addButton(self.ssavep_radio)

        vertical_radio_layout.addWidget(self.n_back_radio)
        vertical_radio_layout.addWidget(self.arithmetic_task_radio)
        vertical_radio_layout.addWidget(self.ssavep_radio)

        # 将垂直布局添加到水平布局中
        horizontal_layout.addLayout(vertical_radio_layout)

        # 将水平布局添加到主布局中
        main_layout.addLayout(horizontal_layout)

        self.next_button = QPushButton('下一步', self)
        self.next_button.clicked.connect(self.open_parameter_config)
        main_layout.addWidget(self.next_button)

        self.setLayout(main_layout)

    def open_parameter_config(self):
        # 获取当前选择的单选按钮的文本
        selected_experiment = self.radio_button_group.checkedButton().text()

        self.param_config_window = DefaultParameterConfigWindow(selected_experiment)
        self.param_config_window.show()  # 启动新窗口
        self.close()  # 关闭当前窗口


class DefaultParameterConfigWindow(QWidget):
    def __init__(self, experiment_type):
        super().__init__()

        self.setWindowTitle('配置参数')
        self.setGeometry(200, 200, 400, 400)
        set_window_to_center(self)
        set_icon(self)

        self.experiment_type = experiment_type
        self.edit_parameters = {}

        self.init_ui()

    def init_ui(self):
        """初始化参数配置窗口UI"""
        main_layout = QVBoxLayout()

        # 设置实验类型各自的参数
        # run_experiment_func: 指向对应范式试验的函数；path: 范式试验对应的可自定义的实验参数
        if self.experiment_type == TYPE_N_BACK:
            # config path
            path = 'configs/n_back.yaml'
            run_experiment_func = run_n_back
        elif self.experiment_type == TYPE_Arithmetic_Task:
            path = 'configs/arithmetic_task.yaml'
            run_experiment_func = run_arithmetic_task
        elif self.experiment_type == TYPE_SSAVEP:
            path = 'configs/ssavep.yaml'
            run_experiment_func = run_ssavep
        else:
            raise ValueError(f'Unknown experimental type: {self.experiment_type}')

        if path:
            with open(path, 'r') as file:
                configs = yaml.safe_load(file)
            self.parameters = configs['configs']
        else:
            self.parameters = {}

        layout = QGridLayout()
        row = 0  # 表头在第0行，参数从第1行开始
        # 表头
        header_labels = ["Parameter Name", "Value", "Unit"]
        for col, text in enumerate(header_labels):
            label = QLabel(text)
            label.setFixedHeight(30)
            layout.addWidget(label, row, col)

        row = row + 1
        # 让默认参数变得可修改
        for param_name, param_info in self.parameters.items():
            value = param_info['value']
            unit = param_info.get('unit', '')
            param_type = param_info.get('type', 'single')

            param_label = QLabel(param_name.title(), self)
            if param_type == 'array':  # 处理数组
                value_edit = QTextEdit(self)
                value_edit.setText(", ".join(value))
                self.edit_parameters[param_name] = (value_edit, 'array')
                layout.addWidget(value_edit, row, 1)  # 添加到布局
            elif param_type == 'checkbox':  # 处理复选框
                checkbox_layout = QHBoxLayout()
                checkboxes = []
                for option in value:
                    checkbox = QCheckBox(str(option), self)
                    checkbox_layout.addWidget(checkbox)
                    checkboxes.append(checkbox)
                checkbox_widget = QWidget()
                checkbox_widget.setLayout(checkbox_layout)
                layout.addWidget(checkbox_widget, row, 1)  # 添加到布局
                self.edit_parameters[param_name] = (checkboxes, 'checkbox')
            else:  # 处理单个值
                value_edit = QLineEdit(self)
                value_edit.setText(str(value))
                value_edit.setFixedWidth(100)
                self.edit_parameters[param_name] = (value_edit, 'single')
                layout.addWidget(value_edit, row, 1)  # 添加到布局

            unit_label = QLabel(unit, self)

            layout.addWidget(param_label, row, 0)
            layout.addWidget(unit_label, row, 2)
            row += 1

        main_layout.addLayout(layout)

        self.start_button = QPushButton('启动实验', self)
        # 使用 partial 传递参数
        self.start_button.clicked.connect(partial(self.start_experiment, run_experiment_func))
        main_layout.addWidget(self.start_button)

        self.setLayout(main_layout)

    def start_experiment(self, run_experiment_func):
        """启动选定的实验并传递参数"""
        # params = {key: val.text() for key, val in self.edit_parameters.items()}
        params = {}
        for key, (widget, widget_type) in self.edit_parameters.items():
            if widget_type == 'array':
                text = widget.toPlainText()
                params[key] = text.split(",")
            elif widget_type == 'checkbox':
                selected_options = [int(cb.text()) for cb in widget if cb.isChecked()]
                params[key] = selected_options
            else:
                text = widget.text()
                params[key] = text

        try:
            # start experiment
            run_experiment_func(params)
        except Exception as e:
            QMessageBox.critical(self, '错误', f'启动实验失败: {str(e)}')

        self.close()


def main():
    """主函数，启动应用"""
    app = QApplication(sys.argv)
    ex = ExperimentLauncher()

    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
