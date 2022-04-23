import warnings
from enum import Enum
from functools import partial

from magicgui.widgets import create_widget
from napari.layers import Labels
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
)
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
import pandas as pd
import numpy as np

# adapted from https://github.com/BiAPoL/napari-clusters-plotter/blob/aa7502898c43341c6df5d988d56075778efbc18d/napari_clusters_plotter/_clustering.py#L45

@register_dock_widget(menu="Segmentation post-processing > Object classification (custom properties, APOC)")
class CustomObjectClassifierWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.viewer = napari_viewer

        # widget for the selection of labels layer
        labels_layer_selection_container = QWidget()
        labels_layer_selection_container.setLayout(QHBoxLayout())
        labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
        self.labels_select = create_widget(annotation=Labels, label="labels_layer")
        labels_layer_selection_container.layout().addWidget(self.labels_select.native)

        # widget for the selection of annotation labels layer
        annotation_layer_selection_container = QWidget()
        annotation_layer_selection_container.setLayout(QHBoxLayout())
        annotation_layer_selection_container.layout().addWidget(QLabel("Annotation"))
        self.annotation_select = create_widget(annotation=Labels, label="Annotation")
        annotation_layer_selection_container.layout().addWidget(self.annotation_select.native)

        # widget for the selection of properties to perform clustering
        choose_properties_container = QWidget()
        choose_properties_container.setLayout(QVBoxLayout())
        choose_properties_container.layout().addWidget(QLabel("Measurements"))
        self.properties_list = QListWidget()
        self.properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.properties_list.setGeometry(QRect(10, 10, 101, 291))
        choose_properties_container.layout().addWidget(self.properties_list)

        # Classifier filename
        filename_edit = FileEdit(
            mode=FileDialogMode.OPTIONAL_FILE,
            filter='*.cl',
            value="table_row_classifier.cl")

        # options
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
        options_widget = QWidget()
        options_widget.setLayout(QHBoxLayout())
        options_widget.layout().addWidget(QLabel("Tree depth, num. trees"))
        options_widget.layout().addWidget(num_max_depth_spinner)
        options_widget.layout().addWidget(num_trees_spinner)

        # custom result column name field
        self.custom_name_container = QWidget()
        self.custom_name_container.setLayout(QHBoxLayout())
        self.custom_name_container.layout().addWidget(QLabel("Custom Results Name"))
        self.custom_name = QLineEdit()
        self.custom_name_not_editable = QLineEdit()

        self.custom_name_container.layout().addWidget(self.custom_name)
        self.custom_name_container.layout().addWidget(self.custom_name_not_editable)
        self.custom_name.setPlaceholderText("Custom_random_forest")
        self.custom_name_not_editable.setPlaceholderText("_CLUSTER_ID")
        self.custom_name_not_editable.setReadOnly(True)

        # show_results_as_table
        show_results_as_table = QCheckBox("Show results in properties table")
        show_results_as_table.setChecked(False)

        # show statistics checkbox
        show_classifier_statistics_checkbox = QCheckBox("Show classifier statistics")
        show_classifier_statistics_checkbox.setChecked(False)

        # show correlation matrix
        show_correlation_matrix_checkbox = QCheckBox("Show correlation matrix")
        show_correlation_matrix_checkbox.setChecked(False)

        # Run button
        run_container = QWidget()
        run_container.setLayout(QHBoxLayout())
        run_button = QPushButton("Run")
        run_container.layout().addWidget(run_button)

        # Update measurements button
        update_container = QWidget()
        update_container.setLayout(QHBoxLayout())
        update_button = QPushButton("Update Measurements")
        update_container.layout().addWidget(update_button)

        # adding all widgets to the layout
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(annotation_layer_selection_container)
        self.layout().addWidget(choose_properties_container)
        self.layout().addWidget(update_container)
        self.layout().addWidget(QLabel("Classifier file"))
        self.layout().addWidget(filename_edit.native)
        self.layout().addWidget(options_widget)
        self.layout().addWidget(self.custom_name_container)
        self.layout().addWidget(show_results_as_table)
        self.layout().addWidget(show_classifier_statistics_checkbox)
        self.layout().addWidget(show_correlation_matrix_checkbox)
        self.layout().addWidget(run_container)
        self.layout().setSpacing(0)

        def run_clicked():

            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            if self.annotation_select.value is None:
                warnings.warn("No annotation image was selected!")
                return

            if len(self.properties_list.selectedItems()) == 0:
                warnings.warn("Please select some measurements!")
                return

            self.run(
                self.labels_select.value,
                self.annotation_select.value,
                [i.text() for i in self.properties_list.selectedItems()],
                str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/"),
                num_trees_spinner.value(),
                num_max_depth_spinner.value(),
                self.custom_name.text(),
                show_results_as_table.isChecked(),
                show_classifier_statistics_checkbox.isChecked(),
                show_correlation_matrix_checkbox.isChecked(),
            )

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(self.update_properties_list)

        # update measurements list when a new labels layer is selected
        self.labels_select.changed.connect(self.update_properties_list)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            if item.layout() is not None:
                item.layout().setSpacing(0)
                item.layout().setContentsMargins(3, 3, 3, 3)


    def update_properties_list(self):
        selected_layer = self.labels_select.value

        if selected_layer is not None:
            features = get_layer_tabular_data(selected_layer)
            if features is not None:
                self.properties_list.clear()
                for p in list(features.keys()):
                    if "label" in p or "CLUSTER_ID" in p or "index" in p:
                        continue
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    item.setSelected(True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)
        self.annotation_select.reset_choices(event)

    # this function runs after the run button is clicked
    def run(
        self,
        labels_layer,
        annotation_layer,
        selected_measurements_list,
        classifier_filename,
        num_trees,
        max_depth,
        custom_name,
        show_results_as_table,
        show_classifier_statistics,
        show_correlation_matrix,
    ):
        print("Selected labels layer: " + str(labels_layer))
        print("Selected annotation layer: " + str(annotation_layer))
        print("Selected measurements: " + str(selected_measurements_list))

        features = get_layer_tabular_data(labels_layer)

        # only select the columns the user requested
        selected_properties = features[selected_measurements_list]
        print("selected properties", selected_properties)

        # determine annotation classes
        from skimage.measure import regionprops
        annotation_stats = regionprops(labels_layer.data, intensity_image=annotation_layer.data)

        annotated_classes = np.asarray([s.max_intensity for s in annotation_stats])
        print("annotated classes", annotated_classes)

        import apoc
        classifier = apoc.TableRowClassifier(opencl_filename=classifier_filename, max_depth=max_depth, num_ensembles=num_trees)
        classifier.train(selected_properties, annotated_classes)
        prediction = np.asarray(classifier.predict(selected_properties)).tolist()
        print("RFC predictions finished.")

        # write result back to features/properties of the labels layer
        add_column_to_layer_tabular_data(
            labels_layer, custom_name + "_CLUSTER_ID", prediction
        )

        import pyclesperanto_prototype as cle

        prediced_labels = np.asarray(cle.replace_intensities(labels_layer.data, [0] + prediction))
        self._add_to_viewer(custom_name + "_CLUSTER_ID", prediced_labels)

        if show_results_as_table:
            # show region properties table as a new widget
            from napari_skimage_regionprops import add_table
            add_table(labels_layer, self.viewer)

        if show_classifier_statistics and self.viewer is not None:

            from ._dock_widget import update_model_analysis
            table = QTableWidget()
            update_model_analysis(table, classifier)
            self.viewer.window.add_dock_widget(table, name="Classifier statistics")

        if show_correlation_matrix and self.viewer is not None:
            table = QTableWidget()
            from ._dock_widget import update_table_gui
            correlation_matrix = pd.DataFrame(selected_properties).dropna().corr()

            table.setColumnCount(len(correlation_matrix))
            table.setRowCount(len(correlation_matrix))

            update_table_gui(table, correlation_matrix, minimum_value=-1, maximum_value=1)
            self.viewer.window.add_dock_widget(table, name="Correlation matrix")

    def _add_to_viewer(self, name, data):
        try:
            self.viewer.layers[name].data = data.astype(int)
            self.viewer.layers[name].visible = True
        except KeyError:
            self.viewer.add_labels(data.astype(int), name=name)

# source https://github.com/BiAPoL/napari-clusters-plotter/blob/aa7502898c43341c6df5d988d56075778efbc18d/napari_clusters_plotter/_utilities.py#L32
def add_column_to_layer_tabular_data(layer, column_name, data):
    if hasattr(layer, "properties"):
        layer.properties[column_name] = data
    if hasattr(layer, "features"):
        layer.features.loc[:, column_name] = data

def get_layer_tabular_data(layer):
    if hasattr(layer, "properties") and layer.properties is not None:
        return pd.DataFrame(layer.properties)
    if hasattr(layer, "features") and layer.features is not None:
        return layer.features
    return None