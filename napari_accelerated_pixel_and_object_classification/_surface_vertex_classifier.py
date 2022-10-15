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
)
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode
import pandas as pd
import numpy as np

@register_dock_widget(menu="Surfaces > Surface vertex classification (custom properties, APOC)")
class SurfaceVertexClassifierWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.viewer = napari_viewer

        # widget for the selection of surface layer
        surface_layer_selection_container = QWidget()
        surface_layer_selection_container.setLayout(QHBoxLayout())
        surface_layer_selection_container.layout().addWidget(QLabel("Surface layer"))
        self.surface_select = create_widget(annotation=Surface, label="surface_layer")
        surface_layer_selection_container.layout().addWidget(self.surface_select.native)

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
        self.layout().addWidget(surface_layer_selection_container)
        self.layout().addWidget(choose_properties_container)
        self.layout().addWidget(update_container)
        self.layout().addWidget(QLabel("Classifier file"))
        self.layout().addWidget(filename_edit.native)
        self.layout().addWidget(options_widget)
        self.layout().addWidget(self.custom_name_container)
        self.layout().addWidget(show_results_as_table)
        self.layout().addWidget(run_container)
        self.layout().setSpacing(0)

        def run_clicked():

            if self.surface_select.value is None:
                warnings.warn("No surface layer was selected!")
                return

            if len(self.properties_list.selectedItems()) == 0:
                warnings.warn("Please select some measurements!")
                return

            self.run(
                self.surface_select.value,
                [i.text() for i in self.properties_list.selectedItems()],
                str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/"),
                num_trees_spinner.value(),
                num_max_depth_spinner.value(),
                self.custom_name.text(),
                show_results_as_table.isChecked(),
            )

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(self.update_properties_list)

        # update measurements list when a new surface layer is selected
        self.surface_select.changed.connect(self.update_properties_list)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            if item.layout() is not None:
                item.layout().setSpacing(0)
                item.layout().setContentsMargins(3, 3, 3, 3)

        self.update_properties_list()

    def update_properties_list(self):
        selected_layer = self.surface_select.value

        if selected_layer is not None:
            from ._custom_table_row_classifier import get_layer_tabular_data
            features = get_layer_tabular_data(selected_layer)
            if features is not None:
                self.properties_list.clear()
                for p in list(features.keys()):
                    if "label" in p or "CLUSTER_ID" in p or "index" in p or "vertex_index" in p:
                        continue
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    item.setSelected(True)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.surface_select.reset_choices(event)

    # this function runs after the run button is clicked
    def run(
        self,
        surface_layer,
        selected_measurements_list,
        classifier_filename,
        num_trees,
        max_depth,
        custom_name,
        show_results_as_table,
    ):
        print("Selected surface layer: " + str(surface_layer))
        print("Selected measurements: " + str(selected_measurements_list))

        from ._custom_table_row_classifier import get_layer_tabular_data
        features = get_layer_tabular_data(surface_layer)

        # only select the columns the user requested
        selected_properties = features[selected_measurements_list]
        print("selected properties", selected_properties)

        # determine annotation classes
        annotated_classes = surface_layer.data[2]
        minimum = np.min(annotated_classes)
        if minimum != 0:
            print("As there are no 0 in the annotated classes, the minimum is subtracted")
            annotated_classes = np.asarray(annotated_classes)
            annotated_classes = annotated_classes - minimum
        print("annotated classes", annotated_classes)

        import apoc
        classifier = apoc.TableRowClassifier(opencl_filename=classifier_filename, max_depth=max_depth, num_ensembles=num_trees)
        classifier.train(selected_properties, annotated_classes)
        prediction = np.asarray(classifier.predict(selected_properties))

        prediction = prediction + minimum
        print("RFC predictions finished.", prediction)

        # write result back to features/properties of the surface layer
        from ._custom_table_row_classifier import add_column_to_layer_tabular_data

        selected_column = custom_name + "_CLUSTER_ID"

        add_column_to_layer_tabular_data(
            surface_layer, selected_column, prediction
        )

        data = surface_layer.data
        data = [np.asarray(data[0]).copy(), np.asarray(data[1]).copy(), prediction]

        new_layer = self.viewer.add_surface(data, name=selected_column)
        new_layer.contrast_limits = list(surface_layer.contrast_limits)
        new_layer.colormap = "hsv"

        if show_results_as_table:
            # show region properties table as a new widget
            from napari_skimage_regionprops import add_table
            add_table(surface_layer, self.viewer)
