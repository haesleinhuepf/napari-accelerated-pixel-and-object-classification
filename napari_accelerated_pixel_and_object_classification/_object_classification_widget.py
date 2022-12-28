from apoc import ObjectClassifier, ObjectSelector
import warnings
from enum import Enum
from functools import partial
from magicgui.widgets import create_widget
from napari.layers import Surface
from napari_tools_menu import register_dock_widget
from qtpy.QtCore import QRect
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSpinBox,
    QCheckBox,
    QTableWidget,
    QScrollArea,
    QPlainTextEdit,
    QSpacerItem,
    QSizePolicy,
    QTabWidget
)
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
import pandas as pd
import numpy as np
from napari.layers import Labels, Image
from ._dock_widget import set_border

@register_dock_widget(menu="Segmentation post-processing > Object classification (APOC)")
class ObjectClassification(QWidget):
    def __init__(self, napari_viewer, classifier_class=ObjectClassifier):

        from ._dock_widget import _add_to_viewer

        super().__init__()
        self.viewer = napari_viewer
        napari_viewer.layers.selection.events.changed.connect(self._on_selection)

        self.classifier_class = classifier_class

        self.setLayout(QVBoxLayout())

        # ----------------------------------------------------------
        # Image + Labels selection
        w, self.image_select = make_widget(annotation=Image, label="Image")
        self.layout().addWidget(w)

        w, self.labels_select = make_widget(annotation=Labels, label="Labels")
        self.layout().addWidget(w)

        # ----------------------------------------------------------
        # Classifier filename
        self.layout().addWidget(QLabel("Classifier file"))
        filename_edit = FileEdit(
            mode=FileDialogMode.OPTIONAL_FILE,
            filter='*.cl',
            value=str(self.classifier_class.__name__) + ".cl")
        self.layout().addWidget(filename_edit.native)

        # ----------------------------------------------------------
        training_widget = QWidget()
        training_widget.setLayout(QVBoxLayout())

        temp = QWidget()
        temp.setLayout(QHBoxLayout())
        postfix = ""
        if self.classifier_class == ObjectSelector:
            postfix = " + class ID to select"

        w, self.annotation_select = make_widget(annotation=Labels, label="annotation" + postfix)
        temp.layout().addWidget(w)

        num_object_annotation_spinner = QSpinBox()
        num_object_annotation_spinner.setToolTip("Please select the label ID / class that should be focused on while training the classifier.")
        num_object_annotation_spinner.setMaximumWidth(40)
        num_object_annotation_spinner.setMinimum(1)
        num_object_annotation_spinner.setValue(2)
        if self.classifier_class == ObjectSelector:
            temp.layout().addWidget(num_object_annotation_spinner)
        training_widget.layout().addWidget(temp)
        set_border(temp)

        num_max_depth_spinner = QSpinBox()
        num_max_depth_spinner.setMinimum(2)
        num_max_depth_spinner.setMaximum(10)
        num_max_depth_spinner.setValue(2)
        num_max_depth_spinner.setToolTip("The more image channels and features you selected, the higher the maximum tree depth should be to retrieve a reliable and robust classifier. The deeper the trees, the longer processing will take.")

        num_trees_spinner = QSpinBox()
        num_trees_spinner.setMinimum(1)
        num_trees_spinner.setMaximum(1000)
        num_trees_spinner.setValue(100)
        num_trees_spinner.setToolTip("The more image channels and features you selected, the more trees should be used to retrieve a reliable and robust classifier. The more trees, the longer processing will take.")

        # Max Depth / Number of ensembles
        temp = QWidget()
        temp.setLayout(QHBoxLayout())
        temp.layout().addWidget(QLabel("Tree depth, num. trees"))
        temp.layout().addWidget(num_max_depth_spinner)
        temp.layout().addWidget(num_trees_spinner)
        training_widget.layout().addWidget(temp)
        set_border(temp)

        # ----------------------------------------------------------
        # Feature selection
        scroll_area = QScrollArea()
        scroll_area.setMinimumWidth(300)
        scroll_area.setMaximumHeight(200)
        scroll_area.setWidgetResizable(True)

        scroll_area_widget = QWidget()
        scroll_area_widget.setLayout(QVBoxLayout())

        minimum_intensity = create_widget(annotation=bool, label="minimum_intensity")
        mean_intensity = create_widget(annotation=bool, label="mean_intensity")
        mean_intensity.value = True
        maximum_intensity = create_widget(annotation=bool, label="maximum_intensity")
        sum_intensity = create_widget(annotation=bool, label="sum_intensity")
        standard_deviation_intensity = create_widget(annotation=bool, label="standard_deviation_intensity")
        pixel_count = create_widget(annotation=bool, label="pixel_count")
        pixel_count.value = True
        shape_extension_ratio = create_widget(annotation=bool, label="shape (extension ratio)")
        centroid_position = create_widget(annotation=bool, label="centroid_position")
        touching_neighbor_count = create_widget(annotation=bool, label="touching_neighbor_count")
        average_centroid_distance_of_touching_neighbors = create_widget(annotation=bool, label="average_centroid_distance_of_touching_neighbors")
        centroid_distance_to_nearest_neighbor = create_widget(annotation=bool, label="centroid_distance_to_nearest_neighbor")
        average_centroid_distance_to_6_nearest_neighbors = create_widget(annotation=bool, label="average_centroid_distance_to_6_nearest_neighbors")
        average_centroid_distance_to_10_nearest_neighbors = create_widget(annotation=bool, label="average_centroid_distance_to_10_nearest_neighbors")
        maximum_distance_of_touching_neighbors = create_widget(annotation=bool, label="maximum_distance_of_touching_neighbors")
        touch_count_sum = create_widget(annotation=bool, label="touch_count_sum")
        minimum_touch_portion = create_widget(annotation=bool, label="minimum_touch_portion")
        standard_deviation_touch_portion = create_widget(annotation=bool, label="standard_deviation_touch_portion")

        scroll_area_widget.layout().addWidget(minimum_intensity.native)
        scroll_area_widget.layout().addWidget(mean_intensity.native)
        scroll_area_widget.layout().addWidget(maximum_intensity.native)
        scroll_area_widget.layout().addWidget(sum_intensity.native)
        scroll_area_widget.layout().addWidget(standard_deviation_intensity.native)
        scroll_area_widget.layout().addWidget(pixel_count.native)
        scroll_area_widget.layout().addWidget(shape_extension_ratio.native)
        scroll_area_widget.layout().addWidget(centroid_position.native)
        scroll_area_widget.layout().addWidget(touching_neighbor_count.native)
        scroll_area_widget.layout().addWidget(average_centroid_distance_of_touching_neighbors.native)
        scroll_area_widget.layout().addWidget(centroid_distance_to_nearest_neighbor.native)
        scroll_area_widget.layout().addWidget(average_centroid_distance_to_6_nearest_neighbors.native)
        scroll_area_widget.layout().addWidget(average_centroid_distance_to_10_nearest_neighbors.native)
        scroll_area_widget.layout().addWidget(maximum_distance_of_touching_neighbors.native)
        scroll_area_widget.layout().addWidget(touch_count_sum.native)
        scroll_area_widget.layout().addWidget(minimum_touch_portion.native)
        scroll_area_widget.layout().addWidget(standard_deviation_touch_portion.native)

        scroll_area.setWidget(scroll_area_widget)

        training_widget.layout().addWidget(scroll_area)

        # ----------------------------------------------------------
        # show statistics checkbox
        show_classifier_statistics_checkbox = QCheckBox("Show classifier statistics")
        show_classifier_statistics_checkbox.setChecked(False)
        training_widget.layout().addWidget(show_classifier_statistics_checkbox)

        # Train button
        button = QPushButton("Train")

        def train_clicked(*arg, **kwargs):

            if self.image_select.value is None:
                raise ValueError("No image selected")

            if self.labels_select.value is None:
                raise ValueError("No labels selected")

            if self.annotation_select.value is None:
                raise ValueError("No annotation selected")

            if self.labels_select.value is self.annotation_select.value:
                raise ValueError("Labels and annotation must not be the same")

            from ._function import Train_object_classifier

            filename = str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/")

            result = _train_classifier(
                self.image_select.value.data,
                self.labels_select.value.data,
                self.annotation_select.value.data,
                filename,
                num_max_depth_spinner.value(),
                num_trees_spinner.value(),
                minimum_intensity.value,
                mean_intensity.value,
                maximum_intensity.value,
                sum_intensity.value,
                standard_deviation_intensity.value,
                pixel_count.value,
                shape_extension_ratio.value,
                centroid_position.value,
                touching_neighbor_count.value,
                average_centroid_distance_of_touching_neighbors.value,
                centroid_distance_to_nearest_neighbor.value,
                average_centroid_distance_to_6_nearest_neighbors.value,
                average_centroid_distance_to_10_nearest_neighbors.value,
                maximum_distance_of_touching_neighbors.value,
                touch_count_sum.value,
                minimum_touch_portion.value,
                standard_deviation_touch_portion.value,
                show_classifier_statistics_checkbox.isChecked(),
                viewer=self.viewer,
                classifier_class=self.classifier_class,
                positive_class_identifier=num_object_annotation_spinner.value(),
            )

            short_filename = filename.split("/")[-1]
            _add_to_viewer(self.viewer, False, "Result of " + short_filename, result, self.image_select.value.scale)

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
        button = QPushButton("Apply classifier")

        def predict_clicked(*arg, **kwargs):
            filename = str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/")

            text_area.setPlainText("# python code to apply this " + str(
                                    self.classifier_class.__name__) + "\n"
                                   "import apoc\n\n" +
                                   "classifier = apoc." + str(
                                    self.classifier_class.__name__) + "(opencl_filename='" + filename + "')\n\n" +
                                   "my_result = classifier.predict(labels=my_labels, image=my_image)")

            from ._function import Apply_object_classification
            result = Apply_object_classification(
                self.image_select.value.data,
                self.labels_select.value.data,
                filename
            )

            short_filename = filename.split("/")[-1]
            _add_to_viewer(self.viewer, False, "Result of " + short_filename, result, self.image_select.value.scale)

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

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._on_selection(event)

    def _on_selection(self, event=None):
        self.labels_select.reset_choices(event)
        self.image_select.reset_choices(event)
        self.annotation_select.reset_choices(event)

def make_widget(annotation, label):
    w = QWidget()
    w.setLayout(QHBoxLayout())
    w.layout().addWidget(QLabel(label))

    magic_w = create_widget(annotation=annotation, label=label)
    w.layout().addWidget(magic_w.native)

    set_border(w)

    return w, magic_w

def _train_classifier(image: "napari.types.ImageData",
                        labels : "napari.types.LabelsData",
                        annotation : "napari.types.LabelsData",
                        model_filename : "magicgui.types.PathLike" = "ObjectClassifier.cl",
                        max_depth : int = 2,
                        num_ensembles : int = 100,
                        minimum_intensity: bool = False,
                        mean_intensity: bool = False,
                        maximum_intensity: bool = False,
                        sum_intensity: bool = False,
                        standard_deviation_intensity: bool = False,
                        pixel_count: bool = True,
                        shape_extension_ratio: bool = False,
                        centroid_position:bool = False,
                        touching_neighbor_count:bool = False,
                        average_centroid_distance_of_touching_neighbors:bool = False,
                        centroid_distance_to_nearest_neighbor:bool = False,
                        average_centroid_distance_to_6_nearest_neighbors:bool = False,
                        average_centroid_distance_to_10_nearest_neighbors:bool = False,
                        maximum_distance_of_touching_neighbors:bool = False,
                        touch_count_sum:bool = False,
                        minimum_touch_portion:bool = False,
                        standard_deviation_touch_portion:bool = False,
                        show_classifier_statistics=False,
                        viewer : "napari.Viewer" = None,
                        classifier_class = ObjectClassifier,
                        positive_class_identifier:int = 2
                        ) -> "napari.types.LabelsData":

    features = ","
    if pixel_count:
        features = features + "area,"
    if minimum_intensity:
        features = features + "min_intensity,"
    if mean_intensity:
        features = features + "mean_intensity,"
    if maximum_intensity:
        features = features + "max_intensity,"
    if standard_deviation_intensity:
        features = features + "standard_deviation_intensity,"
    if sum_intensity:
        features = features + "sum_intensity,"
    if shape_extension_ratio:
        features = features + "mean_max_distance_to_centroid_ratio,"
    if centroid_position:
        features = features + "centroid_x,centroid_y,centroid_z,"
    if touching_neighbor_count:
        features = features + "touching_neighbor_count,"
    if average_centroid_distance_of_touching_neighbors:
        features = features + "average_distance_of_touching_neighbors,"
    if centroid_distance_to_nearest_neighbor:
        features = features + "average_distance_of_n_nearest_neighbors=1,"
    if average_centroid_distance_to_6_nearest_neighbors:
        features = features + "average_distance_of_n_nearest_neighbors=6,"
    if average_centroid_distance_to_10_nearest_neighbors:
        features = features + "average_distance_of_n_nearest_neighbors=10,"
    if maximum_distance_of_touching_neighbors:
        features = features + "maximum_distance_of_touching_neighbors,"
    if touch_count_sum:
        features = features + "touch_count_sum,"
    if minimum_touch_portion:
        features = features + "minimum_touch_portion,"
    if standard_deviation_touch_portion:
        features = features + "standard_deviation_touch_portion,"

    # remove first and last comma
    features = features[1:-1]

    if classifier_class == ObjectClassifier:
        clf = classifier_class(opencl_filename=model_filename, num_ensembles=num_ensembles,
                                               max_depth=max_depth)
    elif classifier_class == ObjectSelector:
        clf = classifier_class(opencl_filename=model_filename, num_ensembles=num_ensembles,
                               max_depth=max_depth, positive_class_identifier=positive_class_identifier)
    else:
        raise("Unsupported classifier class")

    clf.train(features, labels, annotation, image)
    result = clf.predict(labels, image)

    if viewer is not None:
        from napari_workflows._workflow import _get_layer_from_data
        layer = _get_layer_from_data(viewer, labels)

        if hasattr(clf, "_data") and layer is not None:
            for key, item in clf._data.items():
                print(key, type(item), item.shape)
            layer.properties = clf._data

        if show_classifier_statistics:
            from ._dock_widget import update_model_analysis
            table = QTableWidget()
            update_model_analysis(table, clf)
            viewer.window.add_dock_widget(table, name="Classifier statistics")


    return result

@register_dock_widget(menu="Segmentation post-processing > Object selection (APOC)")
class ObjectSelection(ObjectClassification):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer, classifier_class=ObjectSelector)