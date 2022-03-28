import warnings
from functools import partial

from magicgui.widgets import create_widget
from magicgui.widgets import FileEdit
from magicgui.types import FileDialogMode

from napari.layers import Labels
from napari_tools_menu import register_dock_widget
from qtpy.QtCore import QRect
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSpinBox
)
import pandas as pd


# Adapted from https://github.com/BiAPoL/napari-clusters-plotter/blob/main/napari_clusters_plotter/_clustering.py
@register_dock_widget(menu="Segmentation post-processing > Object / custom feature classification (APOC)")
class LabelClassificationWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer

        # QVBoxLayout - lines up widgets vertically
        self.setLayout(QVBoxLayout())
        label_container = QWidget()
        label_container.setLayout(QVBoxLayout())

        # widget for the selection of labels layer
        labels_layer_selection_container = QWidget()
        labels_layer_selection_container.setLayout(QHBoxLayout())
        labels_layer_selection_container.layout().addWidget(QLabel("Labels layer"))
        self.labels_select = create_widget(annotation=Labels, label="labels_layer")
        labels_layer_selection_container.layout().addWidget(self.labels_select.native)

        # widget for the selection of labels layer
        annotation_layer_selection_container = QWidget()
        annotation_layer_selection_container.setLayout(QHBoxLayout())
        annotation_layer_selection_container.layout().addWidget(QLabel("Annotation layer"))
        self.annotation_select = create_widget(annotation=Labels, label="annotation_layer")
        annotation_layer_selection_container.layout().addWidget(self.annotation_select.native)

        # select classifier filename
        filename_edit = FileEdit(
            mode=FileDialogMode.OPTIONAL_FILE,
            filter='*.cl',
            value="TableRowClassifier.cl"
        )

        # select properties of which to produce a dimensionality reduced version
        choose_properties_container = QWidget()
        self.properties_list = QListWidget()
        self.properties_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.properties_list.setGeometry(QRect(10, 10, 101, 291))

        choose_properties_container.setLayout(QVBoxLayout())
        choose_properties_container.layout().addWidget(QLabel("Features"))
        choose_properties_container.layout().addWidget(self.properties_list)

        # Update measurements button
        update_container = QWidget()
        update_container.setLayout(QHBoxLayout())
        update_button = QPushButton("Update Features")
        update_container.layout().addWidget(update_button)

        num_max_depth_spinner = QSpinBox()
        num_max_depth_spinner.setMinimum(2)
        num_max_depth_spinner.setMaximum(10)
        num_max_depth_spinner.setValue(2)
        num_max_depth_spinner.setToolTip(
            "The more image channels and features you selected, the higher the maximum tree depth should be to retrieve a reliable and robust classifier. The deeper the trees, the longer processing will take.")

        num_trees_spinner = QSpinBox()
        num_trees_spinner.setMinimum(1)
        num_trees_spinner.setMaximum(1000)
        num_trees_spinner.setValue(10)
        num_trees_spinner.setToolTip(
            "The more image channels and features you selected, the more trees should be used to retrieve a reliable and robust classifier. The more trees, the longer processing will take.")

        # Max Depth / Number of ensembles
        rfc_config_widget = QWidget()
        rfc_config_widget.setLayout(QHBoxLayout())
        rfc_config_widget.layout().addWidget(QLabel("Tree depth, num. trees"))
        rfc_config_widget.layout().addWidget(num_max_depth_spinner)
        rfc_config_widget.layout().addWidget(num_trees_spinner)

        # classifier statistics checkbox
        classifier_statistics_checkbox = QCheckBox("Show classifier statistics")

        # Run button
        run_widget = QWidget()
        run_widget.setLayout(QHBoxLayout())
        run_button = QPushButton("Train")
        run_widget.layout().addWidget(run_button)

        def run_clicked():

            if self.labels_select.value is None:
                warnings.warn("No labels image was selected!")
                return

            if self.annotation_select.value is None:
                warnings.warn("No annotation was selected!")
                return

            if len(self.properties_list.selectedItems()) == 0:
                warnings.warn("Please select some features!")
                return

            self.run(
                self.labels_select.value,
                self.annotation_select.value,
                [i.text() for i in self.properties_list.selectedItems()],
                str(filename_edit.value.absolute()).replace("\\", "/").replace("//", "/"),
                num_max_depth_spinner.value(),
                num_trees_spinner.value(),
                classifier_statistics_checkbox.isChecked()
            )

        run_button.clicked.connect(run_clicked)
        update_button.clicked.connect(self.update_properties_list)

        # update measurements list when a new labels layer is selected
        self.labels_select.changed.connect(self.update_properties_list)

        # adding all widgets to the layout
        self.layout().addWidget(label_container)
        self.layout().addWidget(labels_layer_selection_container)
        self.layout().addWidget(annotation_layer_selection_container)
        self.layout().addWidget(QLabel("Classifier file"))
        self.layout().addWidget(filename_edit.native)
        self.layout().addWidget(choose_properties_container)
        self.layout().addWidget(update_container)
        self.layout().addWidget(rfc_config_widget)
        self.layout().addWidget(classifier_statistics_checkbox)
        self.layout().addWidget(run_widget)
        self.layout().setSpacing(0)

        # go through all widgets and change spacing
        for i in range(self.layout().count()):
            item = self.layout().itemAt(i).widget()
            if item.layout() is not None:
                item.layout().setSpacing(0)
                item.layout().setContentsMargins(3, 3, 3, 3)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.reset_choices()

    def reset_choices(self, event=None):
        self.labels_select.reset_choices(event)
        self.annotation_select.reset_choices(event)

    def update_properties_list(self):
        selected_layer = self.labels_select.value
        if selected_layer is not None:
            features = get_layer_tabular_data(selected_layer)
            if features is not None:
                self.properties_list.clear()
                for p in list(features.keys()):
                    if (
                        "label" in p
                        or "CLUSTER_ID" in p
                        or "index" in p
                    ):
                        continue
                    item = QListWidgetItem(p)
                    self.properties_list.addItem(item)
                    item.setSelected(True)

    # this function runs after the run button is clicked
    def run(
            self,
            labels_layer,
            annotation_layer,
            selected_measurements_list,
            classifier_filename,
            max_depth,
            num_ensembles,
            show_classifier_statistics
    ):
        print("Selected labels layer: " + str(labels_layer))
        print("Selected measurements: " + str(selected_measurements_list))

        import apoc
        import pyclesperanto_prototype as cle
        trc = apoc.TableRowClassifier(classifier_filename, max_depth, num_ensembles)

        # pick the right columns
        data = get_layer_tabular_data(labels_layer)
        dataframe = pd.DataFrame(data)
        filtered_data = dataframe[selected_measurements_list].to_dict(orient='list')

        # determine ground truth
        annotation_statistics = cle.statistics_of_labelled_pixels(annotation_layer.data, labels_layer.data)
        classification_gt = annotation_statistics['max_intensity']

        # train and predict
        trc.train(filtered_data, classification_gt)
        prediction = trc.predict(filtered_data, return_numpy=True).tolist()

        # store prediction in labels layer
        data["RFC_custom_CLUSTER_ID"] = prediction
        labels_layer.properties = data

        # store prediction as new layer
        label_classification = [0] + prediction
        new_labels = cle.replace_intensities(labels_layer.data, label_classification)
        short_filename = classifier_filename.split("/")[-1]
        self._add_to_viewer("Result of " + short_filename, new_labels)

        if show_classifier_statistics and self.viewer is not None:
            from qtpy.QtWidgets import QTableWidget
            from ._dock_widget import update_model_analysis
            table = QTableWidget()
            update_model_analysis(table, trc)
            self.viewer.window.add_dock_widget(table)

    def _add_to_viewer(self, name, data):
        try:
            self.viewer.layers[name].data = data.astype(int)
            self.viewer.layers[name].visible = True
        except KeyError:
            self.viewer.add_labels(data.astype(int), name=name)

        print("Label classification finished")


# copied from https://github.com/BiAPoL/napari-clusters-plotter/blob/main/napari_clusters_plotter/_utilities.py
# necessary as long as layer.properties and layer.features coexist in napari
def get_layer_tabular_data(layer):
    if hasattr(layer, "properties") and layer.properties is not None:
        return pd.DataFrame(layer.properties)
    if hasattr(layer, "features") and layer.features is not None:
        return layer.features
    return None


def set_features(layer, tabular_data):
    if hasattr(layer, "properties"):
        layer.properties = tabular_data
    if hasattr(layer, "features"):
        layer.features = tabular_data

