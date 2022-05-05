import time
import warnings

import apoc
from qtpy.QtWidgets import QSpacerItem, QSizePolicy
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox, QCheckBox
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QTabWidget, QComboBox, QPlainTextEdit
from qtpy.QtCore import Qt
from qtpy.QtGui import QBrush, QColor, QFont
from magicgui.widgets import Table
from napari._qt.qthreading import thread_worker
from qtpy.QtCore import QTimer, QRect
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
from superqt import QCollapsible

import numpy as np
import napari

from apoc import PredefinedFeatureSet, ObjectSegmenter, PixelClassifier, ProbabilityMapper
from napari_tools_menu import register_dock_widget

@register_dock_widget(menu="Segmentation / labeling > Object segmentation (APOC)")
class ObjectSegmentation(QWidget):
    def __init__(self, napari_viewer, classifier_class=ObjectSegmenter):
        super().__init__()
        self.viewer = napari_viewer
        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.classifier_class = classifier_class

        self.current_annotation = None

        self.setLayout(QVBoxLayout())

        # ----------------------------------------------------------
        # Image selection list
        self.layout().addWidget(QLabel("Select images (channels) used for training"))
        self.image_list = QListWidget()
        self.image_list.setToolTip("The selected image[s] will be considered as individual channels of the same scene. These images should be spatially related and must have the same size.")
        self.image_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        #self.image_list.setGeometry(QRect(10, 10, 10, 10))
        self.image_list.setMaximumHeight(100)
        self.update_image_list()
        self.layout().addWidget(self.image_list)

        # ----------------------------------------------------------
        # Classifier filename
        self.layout().addWidget(QLabel("Classifier file"))
        filename_edit = FileEdit(
            mode=FileDialogMode.OPTIONAL_FILE,
            filter='*.cl',
            value=str(self.classifier_class.__name__) + ".cl")
        self.layout().addWidget(filename_edit.native)

        # ----------------------------------------------------------
        # Training
        training_widget = QWidget()
        training_widget.setLayout(QVBoxLayout())

        # Annotation
        if self.classifier_class == ObjectSegmenter:
            suffix = " + object class"
        elif self.classifier_class == ProbabilityMapper:
            suffix = " + class for probability output"
        else:
            suffix = ""
        training_widget.layout().addWidget(QLabel("Select ground truth annotation" + suffix))
        self.label_list = QComboBox()
        self.label_list.setToolTip("Please provide a labels layer with an annotation to guide the Random Forest training. This labels layer must have the same size as the images selected above.")
        self.update_label_list()

        temp = QWidget()
        temp.setLayout(QHBoxLayout())

        temp.layout().addWidget(self.label_list)
        set_border(self.label_list)

        num_object_annotation_spinner = QSpinBox()
        num_object_annotation_spinner.setToolTip("Please select the label ID / class that should be focused on while training the classifier.")
        num_object_annotation_spinner.setMaximumWidth(40)
        num_object_annotation_spinner.setMinimum(1)
        num_object_annotation_spinner.setValue(2)
        if self.classifier_class == ObjectSegmenter:
            temp.layout().addWidget(num_object_annotation_spinner)
        elif self.classifier_class == ProbabilityMapper:
            temp.layout().addWidget(num_object_annotation_spinner)
        set_border(num_object_annotation_spinner)
        training_widget.layout().addWidget(temp)
        set_border(temp)

        # Features
        collabsible = QCollapsible("Select features")
        training_widget.layout().addWidget(collabsible)

        #feature_selection_button = QPushButton("Select features")
        #training_widget.layout().addWidget(feature_selection_button)

        self.feature_selector = FeatureSelector(self, PredefinedFeatureSet.v070.value)
        collabsible.addWidget(self.feature_selector)
        collabsible.setDuration(0)
        set_border(collabsible)
        #training_widget.layout().addWidget(self.feature_selector)
        #@feature_selection_button.clicked.connect
        #def toggle_feature_selector_visibility():
        #    self.feature_selector.setVisible(not self.feature_selector.isVisible())
        #self.feature_selector.setVisible(False)

        num_max_depth_spinner = QSpinBox()
        num_max_depth_spinner.setMinimum(2)
        num_max_depth_spinner.setMaximum(10)
        num_max_depth_spinner.setValue(2)
        num_max_depth_spinner.setToolTip("The more image channels and features you selected, the higher the maximum tree depth should be to retrieve a reliable and robust classifier. The deeper the trees, the longer processing will take.")

        num_trees_spinner = QSpinBox()
        num_trees_spinner.setMinimum(1)
        num_trees_spinner.setMaximum(1000)
        num_trees_spinner.setValue(10)
        num_trees_spinner.setToolTip("The more image channels and features you selected, the more trees should be used to retrieve a reliable and robust classifier. The more trees, the longer processing will take.")

        # Max Depth / Number of ensembles
        temp = QWidget()
        temp.setLayout(QHBoxLayout())
        temp.layout().addWidget(QLabel("Tree depth, num. trees"))
        temp.layout().addWidget(num_max_depth_spinner)
        temp.layout().addWidget(num_trees_spinner)
        training_widget.layout().addWidget(temp)
        set_border(temp)

        self.label_memory_consumption = QLabel("")
        self.label_memory_consumption.setToolTip("Try to keep estimated memory consumption low. This will also speed up computation.")
        training_widget.layout().addWidget(self.label_memory_consumption)

        # show statistics checkbox
        show_classifier_statistics_checkbox = QCheckBox("Show classifier statistics")
        show_classifier_statistics_checkbox.setChecked(False)
        training_widget.layout().addWidget(show_classifier_statistics_checkbox)

        # Train button
        button = QPushButton("Train")
        def train_clicked(*arg, **kwargs):
            if self.get_selected_annotation() is None:
                warnings.warn("No ground truth annotation selected!")
                return

            if not self.check_image_sizes():
                warnings.warn("Selected images and annotation must have the same dimensionality and size!")
                return

            self.train(
                self.get_selected_images_data(),
                self.get_selected_annotation_data(),
                num_object_annotation_spinner.value(),
                self.feature_selector.getFeatures(),
                num_max_depth_spinner.value(),
                num_trees_spinner.value(),
                str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/"),
                show_classifier_statistics_checkbox.isChecked(),
            )
        button.clicked.connect(train_clicked)
        training_widget.layout().addWidget(button)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        training_widget.layout().addItem(verticalSpacer)

        # ----------------------------------------------------------
        # Prediction
        prediction_widget = QWidget()
        prediction_widget.setLayout(QVBoxLayout())

        # code text area
        text_area = QPlainTextEdit()

        # Predict button
        button = QPushButton("Apply classifier / predict segmentation")
        def predict_clicked(*arg, **kwargs):
            filename = str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/")

            image_names = ", ".join(["image" + str(i) for i, j in enumerate(self.get_selected_images())])
            if ", " in image_names:
                image_names = "[" + image_names + "]"
            text_area.setPlainText("# python code to apply this object segmenter\n"
                                   "from apoc import " + str(self.classifier_class.__name__) + "\n\n" +
                                   "segmenter = " + str(self.classifier_class.__name__) + "(opencl_filename='" + filename + "')\n\n" +
                                   "result = segmenter.predict(image=" + image_names + ")")

            self.predict(
                self.get_selected_images_data(),
                filename
            )

        button.clicked.connect(predict_clicked)
        prediction_widget.layout().addWidget(button)

        prediction_widget.layout().addWidget(text_area)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        prediction_widget.layout().addItem(verticalSpacer)

        # Training / Prediction tabs
        tabs = QTabWidget()
        tabs.addTab(training_widget, "Training")
        tabs.addTab(prediction_widget, "Application / Prediction")

        self.layout().addWidget(tabs)
        set_border(training_widget)

        # ----------------------------------------------------------
        # Spacer
        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.layout().addItem(verticalSpacer)

        # ----------------------------------------------------------
        # Timer for updating memory consumption
        self.timer = QTimer()
        self.timer.setInterval(500)

        @self.timer.timeout.connect
        def update_layer(*_):
            self.update_memory_consumption()
            try:
                if not self.isVisible():
                    self.timer.stop()
            except RuntimeError:
                self.timer.stop()

        self.timer.start()


    def train(self,
              images,
              annotation,
              object_annotation_value,
              feature_definition,
              num_max_depth,
              num_trees,
              filename,
              show_classifier_statistics,
    ):
        print("train " + str(self.classifier_class.__name__))
        print("num images", len(images))
        print("object annotation value", object_annotation_value)
        print("features", feature_definition)
        print("depth", num_max_depth)
        print("num trees", num_trees)
        print("file", filename)

        if len(images) == 0:
            warnings.warn("No image[s] selected")
            return

        if annotation is None:
            warnings.warn("No ground truth / annotation selected")
            return

        if len(images) == 1:
            images = images[0]

        apoc.erase_classifier(filename)
        classifier = self.classifier_class(
            opencl_filename=filename,
            num_ensembles=num_trees,
            max_depth=num_max_depth)

        if self.classifier_class == ObjectSegmenter:
            classifier.positive_class_identifier = object_annotation_value
        elif self.classifier_class == ProbabilityMapper:
            classifier.output_probability_of_class = object_annotation_value

        print("annotation shape", annotation.shape)

        classifier.train(feature_definition, annotation, images)

        print("Training done. Applying model...")

        result = np.asarray(classifier.predict(features=feature_definition, image=images))

        print("Applying / prediction done.")

        short_filename = filename.split("/")[-1]
        self._add_to_viewer("Result of " + short_filename, result)

        if show_classifier_statistics and self.viewer is not None:
            table = QTableWidget()
            update_model_analysis(table, classifier)
            self.viewer.window.add_dock_widget(table)

    def _add_to_viewer(self, name, data):
        try:
            self.viewer.layers[name].data = data.astype(int)
            self.viewer.layers[name].visible = True
        except KeyError:
            if self.classifier_class == ProbabilityMapper:
                self.viewer.add_image(data, name=name)
            else:
                self.viewer.add_labels(data.astype(int), name=name)

    def predict(self, images, filename):
        print("predict")
        print("num images", len(images))
        print("file", filename)

        if len(images) == 0:
            warnings.warn("No image[s] selected")
            return


        if len(images) == 1:
            images = images[0]

        clf = self.classifier_class(opencl_filename=filename)

        result = np.asarray(clf.predict(image=images))

        print("Applying / prediction done.")

        short_filename = filename.split("/")[-1]

        self._add_to_viewer("Result of " + short_filename, result)

    def update_memory_consumption(self):
        number_of_pixels = np.sum(tuple([np.prod(i.shape) for i in self.get_selected_images_data()]))
        number_of_features = len(self.feature_selector.getFeatures().split(" "))
        number_of_bytes_per_pixel = 4

        bytes = number_of_pixels * number_of_bytes_per_pixel * number_of_features
        text = "{bytes:.1f} MBytes".format(bytes=bytes / 1024 / 1024)
        try:
            self.label_memory_consumption.setText("Estimated memory consumption (GPU): " + text)
        except RuntimeError:
            pass

    def update_label_list(self):
        selected_layer = self.get_selected_annotation()
        selected_index = -1

        self._available_labels = []
        self.label_list.clear()
        i = 0
        for l in self.viewer.layers:
            if isinstance(l, napari.layers.Labels):
                self._available_labels.append(l)
                if l == selected_layer:
                    selected_index = i
                suffix = ""
                if len(l.data.shape) == 4:
                    suffix = " (current timepoint)"
                self.label_list.addItem(l.name + suffix)
                i = i + 1
        self.label_list.setCurrentIndex(selected_index)

    def check_image_sizes(self):
        labels = self.get_selected_annotation_data()
        for image in self.get_selected_images_data():
            if not np.array_equal(image.shape, labels.shape):
                return False
        return True

    def get_selected_annotation(self):
        index = self.label_list.currentIndex()
        if index >= 0:
            return self._available_labels[index]
        return None

    def get_selected_annotation_data(self):
        value = self.get_selected_annotation()
        if value is None:
            return None
        value = value.data
        if len(value.shape) == 4:
            current_time = self.viewer.dims.current_step[0]
            value = value[current_time]
            if value.shape[0] == 1:
                value = value[0]
        return value

    def update_image_list(self):
        selected_images = self.get_selected_images()

        self._available_images = []
        self.image_list.clear()
        for l in self.viewer.layers:
            if isinstance(l, napari.layers.Image):
                suffix = ""
                if len(l.data.shape) == 4:
                    suffix = " (current timepoint)"
                item = QListWidgetItem(l.name + suffix)
                self._available_images.append(l)
                self.image_list.addItem(item)
                if l in selected_images:
                    item.setSelected(True)

        selected_images = self.get_selected_images()
        
    def get_selected_images(self):
        images = []
        if not hasattr(self, "_available_images"):
            return images
        for i, image in enumerate(self._available_images):
            item = self.image_list.item(i)
            if item.isSelected():
                images.append(image)
        return images

    def get_selected_images_data(self):
        image_layers = self.get_selected_images()

        images = []
        for layer in image_layers:
            value = layer.data
            if len(value.shape) == 4:
                current_time = self.viewer.dims.current_step[0]
                value = value[current_time]
                if value.shape[0] == 1:
                    value = value[0]
            images.append(value)
        return images

    def _on_selection(self, event=None):
        num_labels_in_viewer = len([l for l in self.viewer.layers if isinstance(l, napari.layers.Labels)])
        if num_labels_in_viewer != self.label_list.size():
            self.update_label_list()

        num_images_in_viewer = len([l for l in self.viewer.layers if isinstance(l, napari.layers.Image)])
        if num_images_in_viewer != self.image_list.size():
            self.update_image_list()

def update_model_analysis(statistics_table, classifier):
    table, _ = classifier.statistics()
    try:
        statistics_table.setColumnCount(len(next(iter(table.values()))))
        statistics_table.setRowCount(len(table))
    except StopIteration:
        pass
    update_table_gui(statistics_table, table)

def update_table_gui(statistics_table, table, minimum_value=0.0, maximum_value=1.0):
    statistics_table.clear()


    for i, column in enumerate(table.keys()):
        statistics_table.setVerticalHeaderItem(i, QTableWidgetItem(str(i + 1) + " " + column))

        for j, value in enumerate(table.get(column)):
            item = QTableWidgetItem("{:.3f}".format(value))
            if not np.isnan(value):
                brush = QBrush()
                rel_value = (value - minimum_value) / (maximum_value - minimum_value)
                brush.setColor(QColor(int((1.0 - rel_value) * 255), int(rel_value * 255), int((1.0 - rel_value) * 255), 255 ))
                item.setBackground(brush.color())
                item.setForeground(QColor(0,0,0,255))
            statistics_table.setItem(i, j, item)
        statistics_table.setColumnWidth(i, 60)



@register_dock_widget(menu="Segmentation / labeling > Semantic segmentation (APOC)")
class SemanticSegmentation(ObjectSegmentation):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer, classifier_class=PixelClassifier)

@register_dock_widget(menu="Filtering > Probability mapper (APOC)")
class ProbabilityMapping(ObjectSegmentation):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer, classifier_class=ProbabilityMapper)


class FeatureSelector(QWidget):
    def __init__(self, parent, feature_definition:str):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.feature_definition = " " + feature_definition.lower() + " "

        self.available_features = ["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur", "sobel_of_gaussian_blur"]
        self.available_features_short_names = ["Gauss", "DoG", "LoG", "SoG"]
        self.available_features_tool_tips = ["Gaussian filter", "Difference of Gaussian", "Laplacian of Gaussian", "Sobel of Gaussian\nalso known as Gradient Magnitude of Gaussian"]

        self.radii = [0.3, 0.5, 1, 2, 3, 4, 5, 10, 15, 25]

        # Headline
        table = QWidget()
        table.setLayout(QGridLayout())
        label_sigma = QLabel("sigma")
        sigma_help = "Increase sigma in case a pixels classification depends on the intensity of other more proximal pixels."
        label_sigma.setToolTip(sigma_help)
        table.layout().addWidget(label_sigma, 0, 0)
        set_border(table)

        for i, r in enumerate(self.radii):
            label_sigma = QLabel(str(r))
            label_sigma.setToolTip(sigma_help)
            table.layout().addWidget(label_sigma, 0, i + 1)

        # Feature lines
        row = 1
        for f, f_short, f_tooltip in zip(self.available_features, self.available_features_short_names, self.available_features_tool_tips):
            label = QLabel(f_short)
            label.setToolTip(f_tooltip)
            table.layout().addWidget(label, row, 0)
            for i, r in enumerate(self.radii):
                table.layout().addWidget(self._make_checkbox("", f + "=" + str(r), (f + "=" + str(r)) in self.feature_definition), row, i + 1)
            row = row + 1

        self.layout().addWidget(table)

        self.layout().addWidget(self._make_checkbox("Consider original image as well", "original", " original " in self.feature_definition))
        set_border(self)


    def _make_checkbox(self, title, feature, checked):
        checkbox = QCheckBox(title)
        checkbox.setChecked(checked)

        def check_the_box(*args, **kwargs):
            if checkbox.isChecked():
                self._add_feature(feature)
            else:
                self._remove_feature(feature)

        checkbox.stateChanged.connect(check_the_box)
        return checkbox

    def _remove_feature(self, feature):
        self.feature_definition = " " + (self.feature_definition.replace(" " + feature + " ", " ")).strip() + " "
        print(self.feature_definition)

    def _add_feature(self, feature):
        print("adding: " + feature)
        self.feature_definition = self.feature_definition + " " + feature + " "
        print(self.feature_definition)

    def getFeatures(self):
        return self.feature_definition.replace("  ", " ").strip(" ")




@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    from ._function import Train_object_classifier
    from ._custom_table_row_classifier import CustomObjectClassifierWidget
    return [ObjectSegmentation, SemanticSegmentation, Train_object_classifier, CustomObjectClassifierWidget]


def set_border(widget:QWidget, spacing=2, margin=0):
    if hasattr(widget.layout(), "setContentsMargins"):
        widget.layout().setContentsMargins(margin, margin, margin, margin)
    if hasattr(widget.layout(), "setSpacing"):
        widget.layout().setSpacing(spacing)

