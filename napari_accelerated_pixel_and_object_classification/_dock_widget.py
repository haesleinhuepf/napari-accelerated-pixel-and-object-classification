import time
import warnings

import apoc
from qtpy.QtWidgets import QSpacerItem, QSizePolicy
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QSpinBox, QCheckBox
from qtpy.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QGridLayout, QPushButton, QFileDialog
from qtpy.QtWidgets import QListWidget, QListWidgetItem, QAbstractItemView, QTabWidget, QComboBox, QPlainTextEdit
from qtpy.QtCore import Qt
from magicgui.widgets import Table
from napari._qt.qthreading import thread_worker
from qtpy.QtCore import QTimer, QRect
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode

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
        self.layout().addWidget(QLabel("Select image[s]/channels[s] used for training"))
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.image_list.setGeometry(QRect(10, 10, 101, 291))
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
        self.update_label_list()

        temp = QWidget()
        temp.setLayout(QHBoxLayout())

        temp.layout().addWidget(self.label_list)

        num_object_annotation_spinner = QSpinBox()
        num_object_annotation_spinner.setMaximumWidth(40)
        num_object_annotation_spinner.setMinimum(1)
        num_object_annotation_spinner.setValue(2)
        if self.classifier_class == ObjectSegmenter:
            temp.layout().addWidget(num_object_annotation_spinner)
        elif self.classifier_class == ProbabilityMapper:
            temp.layout().addWidget(num_object_annotation_spinner)

        training_widget.layout().addWidget(temp)

        # Features
        training_widget.layout().addWidget(QLabel("Select features"))
        self.feature_selector = FeatureSelector(PredefinedFeatureSet.small_dog_log.value)
        training_widget.layout().addWidget(self.feature_selector)

        num_max_depth_spinner = QSpinBox()
        num_max_depth_spinner.setMinimum(2)
        num_max_depth_spinner.setMaximum(10)
        num_max_depth_spinner.setValue(2)

        num_trees_spinner = QSpinBox()
        num_trees_spinner.setMinimum(1)
        num_trees_spinner.setMaximum(1000)
        num_trees_spinner.setValue(10)

        # Max Depth / Number of ensembles
        temp = QWidget()
        temp.setLayout(QHBoxLayout())
        temp.layout().addWidget(QLabel("Tree depth, num. trees"))
        temp.layout().addWidget(num_max_depth_spinner)
        temp.layout().addWidget(num_trees_spinner)
        training_widget.layout().addWidget(temp)

        self.label_memory_consumption = QLabel("")
        training_widget.layout().addWidget(self.label_memory_consumption)

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
                str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/")
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


    def train(self, images, annotation, object_annotation_value, feature_definition, num_max_depth, num_trees, filename):
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
        clf = self.classifier_class(
            opencl_filename=filename,
            num_ensembles=num_trees,
            max_depth=num_max_depth)

        if self.classifier_class == ObjectSegmenter:
            clf.positive_class_identifier = object_annotation_value
        elif self.classifier_class == ProbabilityMapper:
            clf.output_probability_of_class = object_annotation_value

        print("annotation shape", annotation.shape)

        clf.train(feature_definition, annotation, images)

        print("Training done. Applying model...")

        result = np.asarray(clf.predict(features=feature_definition, image=images))

        print("Applying / prediction done.")

        short_filename = filename.split("/")[-1]
        self._add_to_viewer("Result of " + short_filename, result)

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
        print("selected images was:", selected_images)

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
        print("selected images is:", selected_images)

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

@register_dock_widget(menu="Segmentation / labeling > Semantic segmentation (APOC)")
class SemanticSegmentation(ObjectSegmentation):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer, classifier_class=PixelClassifier)

@register_dock_widget(menu="Filtering > Probability mapper (APOC)")
class ProbabilityMapping(ObjectSegmentation):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer, classifier_class=ProbabilityMapper)


class FeatureSelector(QWidget):
    def __init__(self, feature_definition:str):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.feature_definition = " " + feature_definition.lower() + " "

        self.available_features = ["gaussian_blur", "difference_of_gaussian", "laplace_box_of_gaussian_blur"]
        self.available_features_short_names = ["Gauss", "DoG", "LoG"]

        self.radii = [0.3, 0.5, 1, 2, 3, 4, 5, 10, 15, 25]

        # Headline
        table = QWidget()
        table.setLayout(QGridLayout())
        table.layout().addWidget(QLabel("sigma"), 0, 0)
        table.layout().setSpacing(0)
        if hasattr(table.layout(), "setMargin"):
            table.layout().setMargin(0)

        for i, r in enumerate(self.radii):
            table.layout().addWidget(QLabel(str(r)), 0, i + 1)

        # Feature lines
        row = 1
        for f, f_short in zip(self.available_features, self.available_features_short_names):
            table.layout().addWidget(QLabel(f_short), row, 0)
            for i, r in enumerate(self.radii):
                table.layout().addWidget(self._make_checkbox("", f + "=" + str(r), (f + "=" + str(r)) in self.feature_definition), row, i + 1)
            row = row + 1

        self.layout().addWidget(table)

        self.layout().addWidget(self._make_checkbox("Consider original image as well", "original", " original " in self.feature_definition))



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

@register_dock_widget(menu="Segmentation post-processing > Object classification (APOC)")
class ObjectClassifier(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        from ._function import Train_object_classifier
        napari_viewer.window.add_function_widget(Train_object_classifier)

@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [ObjectSegmentation, SemanticSegmentation, ObjectClassifier]
