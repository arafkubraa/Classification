import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QTextEdit, QMessageBox, QLineEdit
)
from PySide6.QtGui import QPainter, QColor, QPen, QAction, QImage, QFont, QBrush
from PySide6.QtCore import Qt, QRect, QRectF, QPointF, QPoint
import subprocess
import importlib
import my_process_module


GUI_BACKGROUND_COLOR = "#faf8f9"
CONTROL_PANEL_BG_COLOR = "#faf8f9"
CONTROL_GROUP_BG_COLOR = "#ffffff"
LOG_BG_COLOR = "#ffffff"
LOG_TEXT_COLOR = "#212529"
PLOT_AREA_BG_COLOR = "#faf7f2"


PASTEL_COLORS_DICT = {
    1: QColor(244, 143, 177), 2: QColor(120, 200, 160), 3: QColor(135, 179, 255),
    4: QColor(255, 224, 130), 5: QColor(200, 170, 255), 6: QColor(77, 208, 225),
}
PASTEL_REGION_COLORS_DICT = {
    1: QColor(255, 220, 235, 255), 2: QColor(220, 245, 225, 255), 3: QColor(225, 235, 255, 255),
    4: QColor(255, 248, 225, 255), 5: QColor(240, 230, 255, 255), 6: QColor(204, 247, 255, 255),
}


class PlottingWidget(QWidget):
    def __init__(self, parent_form1=None):
        super().__init__(parent_form1)
        self.form1_reference = parent_form1
        self.setFixedSize(802, 578)
        self.setStyleSheet(f"background-color: {PLOT_AREA_BG_COLOR}; border: 1px solid #aaa;")
        self.samples_to_draw = []
        self.decision_boundary_image = None
        self.is_normalized_space = False
        self.epoch_errors = np.array([])

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.decision_boundary_image:
            painter.drawImage(QRect(0, 0, self.width(), self.height()), self.decision_boundary_image)
        cx, cy = self.width() // 2, self.height() // 2
        pen = QPen(QColor(Qt.GlobalColor.black), 1.0)
        painter.setPen(pen)
        painter.drawLine(cx, 0, cx, self.height())
        painter.drawLine(0, cy, self.width(), cy)
        for x_screen, y_screen, label in self.samples_to_draw:
            self._draw_sample(painter, x_screen, y_screen, int(label))
        self._draw_error_graph(painter)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.is_normalized_space and self.form1_reference:
                self.form1_reference.handle_mouse_click(event.position())
            elif self.is_normalized_space:
                QMessageBox.information(self, "Warning", "Cannot add new data in normalized space.")

    def update_samples_to_draw(self, samples_data, targets_data, is_normalized=False):
        self.is_normalized_space = is_normalized
        if not isinstance(samples_data, np.ndarray) or samples_data.size == 0:
            self.samples_to_draw = []
            self.update()
            return
        cx, cy = self.width() // 2, self.height() // 2
        new_list = []
        N = samples_data.shape[0]
        SCALE_FACTOR = 80
        for i in range(N):
            x_data, y_data = samples_data[i, 0], samples_data[i, 1]
            label = targets_data[i]
            x_screen = int(x_data * SCALE_FACTOR + cx) if is_normalized else int(x_data + cx)
            y_screen = int(cy - y_data * SCALE_FACTOR) if is_normalized else int(cy - y_data)
            new_list.append((x_screen, y_screen, label))
        self.samples_to_draw = new_list
        self.update()

    def update_error_graph(self, errors):
        self.epoch_errors = errors
        self.update()

    def _draw_sample(self, painter, x, y, label):
        pen = QPen(Qt.GlobalColor.black, 2.0)
        painter.setPen(pen)
        fill_color = PASTEL_COLORS_DICT.get(label, Qt.GlobalColor.gray)
        brush = QBrush(fill_color)
        painter.setBrush(brush)
        size = 8
        if label == 1:
            size = 6
            heart_points = [
                QPoint(x, y + size), QPoint(x - size, y - int(size * 0.2)), QPoint(x - size, y - size),
                QPoint(x - int(size * 0.5), y - int(size * 1.5)), QPoint(x, y - size), QPoint(x + int(size * 0.5), y - int(size * 1.5)),
                QPoint(x + size, y - size), QPoint(x + size, y - int(size * 0.2)),
            ]
            painter.drawPolygon(heart_points)
        elif label == 2:
            points = [QPointF(x, y - size), QPointF(x + size, y), QPointF(x, y + size), QPointF(x - size, y)]
            painter.drawPolygon(points)
        elif label == 3:
            points = [QPointF(x, y - size), QPointF(x + size, y + size), QPointF(x - size, y + size)]
            painter.drawPolygon(points)
        else:
            radius = size * 0.8
            painter.drawEllipse(QPointF(x, y), radius, radius)

    def _draw_error_graph(self, painter):
        if self.epoch_errors.size < 2: return
        GRAPH_W, GRAPH_H, PADDING = 160, 110, 15
        graph_rect = QRect(PADDING, self.height() - GRAPH_H - PADDING, GRAPH_W, GRAPH_H)
        painter.setBrush(QBrush(QColor(255, 255, 255, 180)))
        painter.setPen(Qt.GlobalColor.darkGray)
        painter.drawRoundedRect(graph_rect, 5.0, 5.0)
        painter.setPen(Qt.GlobalColor.black)
        title_font = QFont(); title_font.setPixelSize(12)
        painter.setFont(title_font)
        painter.drawText(QRectF(graph_rect.x() + 5, graph_rect.y() + 5, graph_rect.width() - 10, 15),
                         Qt.AlignmentFlag.AlignLeft, "Training Error (Epochs >)")
        
        num_epochs = len(self.epoch_errors)
        max_error = np.max(self.epoch_errors)
        if max_error <= 0: max_error = 1.0
        
        plot_area_rect = QRect(graph_rect.x() + 5, graph_rect.y() + 20, graph_rect.width() - 10, graph_rect.height() - 25)
        painter.setPen(QPen(QColor(217, 83, 79), 2.0))
        points = [
            (plot_area_rect.left() + int((i / (num_epochs - 1)) * plot_area_rect.width()),
             plot_area_rect.bottom() - int((self.epoch_errors[i] / max_error) * plot_area_rect.height()))
            for i in range(num_epochs)
        ]
        for i in range(num_epochs - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])


class Form1(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flexible MLP")
        self.setGeometry(100, 100, 1150, 650)
        
        self.classification_type = "Multi-Layer"
        self.class_count = 3
        self.numSample = 0
        self.inputDim = 2
        
        self.samples_np = np.empty((0, self.inputDim), dtype=np.float32)
        self.targets_np = np.empty((0,), dtype=np.float32)
        
        
        self.layer_sizes = []
        self.weights = []
        self.biases = []

        self.mean_vec = np.array([0.0, 0.0], dtype=np.float32)
        self.std_vec = np.array([1.0, 1.0], dtype=np.float32)
        self._setup_menu()
        self._setup_ui()
        self.set_net(initial=True)

    def _setup_menu(self):
        menu_bar = self.menuBar()
        process_menu = menu_bar.addMenu("Process")
        training_action = QAction("Training", self); training_action.triggered.connect(self.start_training); process_menu.addAction(training_action)
        testing_action = QAction("Test (Decision Boundary)", self); testing_action.triggered.connect(self.test_button_clicked); process_menu.addAction(testing_action)

    def _setup_ui(self):
        central_widget = QWidget()
        central_widget.setStyleSheet(f"background-color: {GUI_BACKGROUND_COLOR};")
        main_layout = QHBoxLayout(central_widget)
        self.plotting_area = PlottingWidget(parent_form1=self)
        main_layout.addWidget(self.plotting_area, 1)
        
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        group_style = "QGroupBox { background-color: #fff; border: 1px solid #dee2e6; border-radius: 5px; margin-top: 1ex; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 2 6px; color: black; }"
        button_style = "QPushButton { background-color: #c8aaff; color: #2c3e50; border-radius: 6px; border: 1px solid #b098e8; font-size: 13px; font-weight: 600; padding: 6px 10px; } QPushButton:hover { background-color: #d2b9ff; } QPushButton:pressed { background-color: #b99bf5; }"
        label_style = "QLabel { color: black; }"
        combo_box_style = "QComboBox { background-color: white; color: black; border: 1px solid #dee2e6; border-radius: 5px; padding: 4px; } QComboBox::drop-down { border: none; } QComboBox QAbstractItemView { background-color: white; color: black; }"
        line_edit_style = "QLineEdit { background-color: white; color: black; border: 1px solid #dee2e6; border-radius: 5px; padding: 4px; }"

        group_net = QGroupBox("Network Architecture")
        group_net.setStyleSheet(group_style)
        net_layout = QVBoxLayout(group_net)
        
        self.ClassCountBox = QComboBox(); self.ClassCountBox.addItems([str(i) for i in range(2, 8)]); self.ClassCountBox.setCurrentText("3")
        self.ClassCountBox.setStyleSheet(combo_box_style)
        self.ClassificationTypeBox = QComboBox(); self.ClassificationTypeBox.addItems(["Single-Layer", "Multi-Layer"]); self.ClassificationTypeBox.setCurrentText("Multi-Layer")
        self.ClassificationTypeBox.setStyleSheet(combo_box_style)

        self.num_hidden_layers_label = QLabel("Number of Hidden Layers:")
        self.num_hidden_layers_label.setStyleSheet(label_style)
        self.num_hidden_layers_input = QLineEdit("1")
        self.num_hidden_layers_input.setStyleSheet(line_edit_style)

        self.neurons_per_layer_label = QLabel("Neurons per Hidden Layer:")
        self.neurons_per_layer_label.setStyleSheet(label_style)
        self.neurons_per_layer_input = QLineEdit("8")
        self.neurons_per_layer_input.setStyleSheet(line_edit_style)
        
        self.Set_Net = QPushButton("Set Network")
        self.Set_Net.setStyleSheet(button_style)
        
        self.ClassCountBox.currentIndexChanged.connect(self.set_net)
        self.ClassificationTypeBox.currentIndexChanged.connect(self._update_network_ui)
        self.Set_Net.clicked.connect(self.set_net)
        
        output_classes_label = QLabel("Output Classes:")
        output_classes_label.setStyleSheet(label_style)
        net_layout.addWidget(output_classes_label)
        net_layout.addWidget(self.ClassCountBox)
        
        network_type_label = QLabel("Network Type:")
        network_type_label.setStyleSheet(label_style)
        net_layout.addWidget(network_type_label)
        net_layout.addWidget(self.ClassificationTypeBox)
        
        net_layout.addWidget(self.num_hidden_layers_label)
        net_layout.addWidget(self.num_hidden_layers_input)
        net_layout.addWidget(self.neurons_per_layer_label)
        net_layout.addWidget(self.neurons_per_layer_input)

        net_layout.addWidget(self.Set_Net)
        control_layout.addWidget(group_net)
        
        group_data = QGroupBox("Data"); group_data.setStyleSheet(group_style)
        data_layout = QHBoxLayout(group_data)
        self.ClassNoBox = QComboBox()
        self.ClassNoBox.setStyleSheet(combo_box_style)
        self.Reset_Button = QPushButton("Reset"); self.Reset_Button.setStyleSheet(button_style); self.Reset_Button.clicked.connect(self.reset_application)
        
        label_label = QLabel("Label:")
        label_label.setStyleSheet(label_style)
        data_layout.addWidget(label_label); data_layout.addWidget(self.ClassNoBox); data_layout.addWidget(self.Reset_Button)
        control_layout.addWidget(group_data)
        
        self.label3 = QLabel("Samples: 0")
        self.label3.setStyleSheet(label_style)
        control_layout.addWidget(self.label3)
        
        self.textBox1 = QTextEdit(); self.textBox1.setPlaceholderText("Logs...")
        self.textBox1.setStyleSheet("background-color: white; color: black; border: 1px solid #dee2e6; border-radius: 5px;")
        control_layout.addWidget(self.textBox1)
        main_layout.addWidget(control_panel)
        self.setCentralWidget(central_widget)
        self._update_network_ui()

    def _update_network_ui(self):
        is_multilayer = self.ClassificationTypeBox.currentText() == "Multi-Layer"
        self.num_hidden_layers_label.setVisible(is_multilayer)
        self.num_hidden_layers_input.setVisible(is_multilayer)
        self.neurons_per_layer_label.setVisible(is_multilayer)
        self.neurons_per_layer_input.setVisible(is_multilayer)

    def log_message(self, msg): self.textBox1.append(msg)
    def update_sample_count(self, c): self.numSample = c; self.label3.setText(f"Samples: {c}")

    def reset_application(self):
        self.samples_np = np.empty((0, self.inputDim), dtype=np.float32)
        self.targets_np = np.empty((0,), dtype=np.float32)
        self.update_sample_count(0)
        self.plotting_area.decision_boundary_image = None
        self.plotting_area.update_samples_to_draw(np.empty((0, 2)), np.empty((0,)))
        self.plotting_area.update_error_graph(np.array([]))
        self.set_net(initial=True)
        self.log_message("Application reset.")

    def set_net(self, initial=False):
        try:
            self.class_count = int(self.ClassCountBox.currentText())
            self.classification_type = self.ClassificationTypeBox.currentText()
            self.ClassNoBox.clear()
            self.ClassNoBox.addItems([str(i) for i in range(1, self.class_count + 1)])

            if self.classification_type == "Multi-Layer":
                num_hidden_layers_str = self.num_hidden_layers_input.text()
                neurons_per_layer_str = self.neurons_per_layer_input.text()

                if not num_hidden_layers_str.isdigit() or not neurons_per_layer_str.isdigit() or int(num_hidden_layers_str) == 0 or int(neurons_per_layer_str) == 0:
                    QMessageBox.warning(self, "Warning", "Please enter valid, non-zero numbers for hidden layers and neurons.")
                    return
                
                num_hidden_layers = int(num_hidden_layers_str)
                neurons_per_layer = int(neurons_per_layer_str)
                
                hidden_layers = [neurons_per_layer] * num_hidden_layers
                self.layer_sizes = [self.inputDim] + hidden_layers + [self.class_count]

                self.weights, self.biases = [], []
                for i in range(len(self.layer_sizes) - 1):
                    w_size = self.layer_sizes[i] * self.layer_sizes[i+1]
                    b_size = self.layer_sizes[i+1]
                    self.weights.append(my_process_module.init_array_random(w_size))
                    self.biases.append(my_process_module.init_array_random(b_size))
                
                log_msg = f"Network set: MLP with layers {self.layer_sizes}."
            else: # Single-Layer
                self.layer_sizes = [self.inputDim, self.class_count]
                w_size = self.inputDim * self.class_count
                b_size = self.class_count
                self.weights = [my_process_module.init_array_random(w_size)]
                self.biases = [my_process_module.init_array_random(b_size)]
                log_msg = f"Network set: Single-Layer with {self.class_count} classes."

            if not initial: self.log_message(log_msg)
            self.plotting_area.decision_boundary_image = None
            self.draw_samples(is_normalized=False)
            self.plotting_area.update_error_graph(np.array([]))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to set network: {e}")

    def handle_mouse_click(self, position):
        if self.class_count == 0: return
        cx, cy = self.plotting_area.width() // 2, self.plotting_area.height() // 2
        x_data, y_data = position.x() - cx, cy - position.y()
        new_sample = np.array([x_data, y_data], dtype=np.float32)
        label = int(self.ClassNoBox.currentText())
        try:
            self.samples_np = np.append(self.samples_np, new_sample)
            self.targets_np = np.append(self.targets_np, label)
            self.numSample += 1
            self.update_sample_count(self.numSample)
            self.log_message(f"Added point: ({x_data:.1f}, {y_data:.1f}) -> Label {label}")
            self.draw_samples(is_normalized=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add data: {e}")

    def draw_samples(self, is_normalized=False):
        if self.numSample == 0:
            self.plotting_area.update_samples_to_draw(np.empty((0, 2)), np.empty((0,)), is_normalized)
            return

        samples_2d = self.samples_np.reshape(-1, self.inputDim)
        if is_normalized:
            std_safe = np.where(np.abs(self.std_vec) < 1e-6, 1.0, self.std_vec)
            norm_samples = (samples_2d - self.mean_vec) / std_safe
            self.plotting_area.update_samples_to_draw(norm_samples, self.targets_np, True)
        else:
            self.plotting_area.update_samples_to_draw(samples_2d, self.targets_np, False)

    def start_training(self):
        if self.numSample < 2 or not self.weights:
            QMessageBox.warning(self, "Warning", "Collect at least 2 samples and set the network first."); return
        self.log_message("Starting build...")
        try:
            subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], check=True, capture_output=True, text=True)
            importlib.reload(my_process_module)
            self.log_message("Build successful. Starting training...")
        except subprocess.CalledProcessError as e:
            self.log_message(f"Build failed:\n{e.stderr}")
            QMessageBox.critical(self, "Build Error", f"Failed to build C++ module:\n{e.stderr}")
            return

        try:
            (self.weights, self.biases, self.mean_vec, self.std_vec, norm_samples, epoch_errors) = \
                my_process_module.Train_MLP_Multiclass(
                    self.samples_np, self.targets_np, self.layer_sizes,
                    self.weights, self.biases, self.numSample
                )
            
            self.log_message(f"Training complete. Epochs: {len(epoch_errors)}, Final Error: {epoch_errors[-1]:.6f}")
            self.log_message(f"Normalization: Mean={np.round(self.mean_vec, 2)}, Std={np.round(self.std_vec, 2)}")

            if self.classification_type == "Single-Layer":
                self.draw_linear_classifier_lines()
            else:
                self.draw_boundary_lines(is_original_space=False)
            
            self.plotting_area.update_samples_to_draw(norm_samples.reshape(-1, self.inputDim), self.targets_np, is_normalized=True)
            self.plotting_area.update_error_graph(epoch_errors)
        except Exception as e:
            self.log_message(f"Training failed: {e}")
            QMessageBox.critical(self, "Training Error", f"An error occurred during training: {e}")

    def test_button_clicked(self):
        self.draw_boundary_regions(is_original_space=True)

    def draw_linear_classifier_lines(self):
        if self.numSample < 2 or not self.weights:
            QMessageBox.warning(self, "Warning", "Need at least 2 samples and a trained network."); return
        self.log_message("Drawing linear classifier lines for single-layer model...")

        W, H = self.plotting_area.width(), self.plotting_area.height()
        cx, cy = W // 2, H // 2
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)

        weights_flat = self.weights[0]
        biases = self.biases[0]
        SCALE_FACTOR = 80

        def draw_line(w0, w1, b, pen):
            painter.setPen(pen)
            if abs(w1) > 1e-6:
                x_norm1 = -W / (2 * SCALE_FACTOR)
                y_norm1 = (-w0 * x_norm1 - b) / w1
                x_norm2 = W / (2 * SCALE_FACTOR)
                y_norm2 = (-w0 * x_norm2 - b) / w1
                p1 = QPointF(x_norm1 * SCALE_FACTOR + cx, cy - y_norm1 * SCALE_FACTOR)
                p2 = QPointF(x_norm2 * SCALE_FACTOR + cx, cy - y_norm2 * SCALE_FACTOR)
                painter.drawLine(p1, p2)
            elif abs(w0) > 1e-6:
                x_norm = -b / w0
                p1 = QPointF(x_norm * SCALE_FACTOR + cx, 0)
                p2 = QPointF(x_norm * SCALE_FACTOR + cx, H)
                painter.drawLine(p1, p2)

        if self.class_count == 2:
            
            w0_c1 = weights_flat[0 * self.inputDim + 0]
            w1_c1 = weights_flat[0 * self.inputDim + 1]
            b_c1 = biases[0]
            w0_c2 = weights_flat[1 * self.inputDim + 0]
            w1_c2 = weights_flat[1 * self.inputDim + 1]
            b_c2 = biases[1]

            w0_diff = w0_c1 - w0_c2
            w1_diff = w1_c1 - w1_c2
            b_diff = b_c1 - b_c2
            
            pen = QPen(Qt.GlobalColor.black, 2.0)
            draw_line(w0_diff, w1_diff, b_diff, pen)
        else:
            
            for c in range(self.class_count):
                w0 = weights_flat[c * self.inputDim + 0]
                w1 = weights_flat[c * self.inputDim + 1]
                b = biases[c]
                color = PASTEL_COLORS_DICT.get(c + 1, QColor(Qt.GlobalColor.black))
                pen = QPen(color, 2.0)
                draw_line(w0, w1, b, pen)

        painter.end()
        self.plotting_area.decision_boundary_image = image
        self.plotting_area.update()
        self.log_message("Linear boundaries drawn.")

    def draw_boundary_regions(self, is_original_space=True):
        if self.numSample < 2 or not self.weights:
            QMessageBox.warning(self, "Warning", "Need at least 2 samples and a trained network."); return
        self.log_message(f"Calculating decision boundary regions... (Space: {'Original' if is_original_space else 'Normalized'})")
        
        W, H = self.plotting_area.width(), self.plotting_area.height()
        cx, cy = W // 2, H // 2
        image = QImage(W, H, QImage.Format.Format_RGB32); image.fill(QColor(PLOT_AREA_BG_COLOR))
        
        std_safe = np.where(np.abs(self.std_vec) < 1e-6, 1.0, self.std_vec)
        
        for row in range(H):
            for col in range(W):
                x_data, y_data = float(col - cx), float(cy - row)
                if is_original_space:
                    x_norm = (x_data - self.mean_vec[0]) / std_safe[0]
                    y_norm = (y_data - self.mean_vec[1]) / std_safe[1]
                else:
                    x_norm, y_norm = x_data / 80.0, y_data / 80.0
                
                test_sample = np.array([x_norm, y_norm], dtype=np.float32)
                predicted_class = my_process_module.Predict_MLP_Multiclass(
                    test_sample, self.layer_sizes, self.weights, self.biases
                )
                color = PASTEL_REGION_COLORS_DICT.get(predicted_class + 1, QColor(PLOT_AREA_BG_COLOR))
                image.setPixelColor(col, row, color)
        
        self.plotting_area.decision_boundary_image = image
        if is_original_space: self.draw_samples(is_normalized=False)
        self.plotting_area.update()
        self.log_message("Boundary regions drawn.")

    def draw_boundary_lines(self, is_original_space=True):
        if self.numSample < 2 or not self.weights:
            QMessageBox.warning(self, "Warning", "Need at least 2 samples and a trained network."); return
        self.log_message(f"Calculating decision boundary lines... (Space: {'Original' if is_original_space else 'Normalized'})")

        W, H = self.plotting_area.width(), self.plotting_area.height()
        cx, cy = W // 2, H // 2
        image = QImage(W, H, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)

        std_safe = np.where(np.abs(self.std_vec) < 1e-6, 1.0, self.std_vec)

        predictions = np.zeros((H, W), dtype=int)
        for row in range(H):
            for col in range(W):
                x_data, y_data = float(col - cx), float(cy - row)
                if is_original_space:
                    x_norm = (x_data - self.mean_vec[0]) / std_safe[0]
                    y_norm = (y_data - self.mean_vec[1]) / std_safe[1]
                else:
                    x_norm, y_norm = x_data / 80.0, y_data / 80.0
                
                test_sample = np.array([x_norm, y_norm], dtype=np.float32)
                predicted_class_idx = my_process_module.Predict_MLP_Multiclass(
                    test_sample, self.layer_sizes, self.weights, self.biases
                )
                predictions[row, col] = predicted_class_idx + 1

        for row in range(1, H):
            for col in range(1, W):
                current_class = predictions[row, col]
                left_class = predictions[row, col - 1]
                top_class = predictions[row - 1, col]

                if current_class != left_class or current_class != top_class:
                    line_color = PASTEL_COLORS_DICT.get(current_class, QColor(Qt.GlobalColor.black))
                    image.setPixelColor(col, row, line_color)

        self.plotting_area.decision_boundary_image = image
        if is_original_space: self.draw_samples(is_normalized=False)
        self.plotting_area.update()
        self.log_message("Boundary lines drawn.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Form1()
    w.show()
    sys.exit(app.exec())