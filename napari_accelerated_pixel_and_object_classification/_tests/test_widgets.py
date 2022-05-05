import numpy as np

import napari_accelerated_pixel_and_object_classification
import pytest

# this is your plugin name declared in your napari.plugins entry point
MY_PLUGIN_NAME = "napari-accelerated-pixel-and-object-classification"
# the name of your widget(s)
MY_WIDGET_NAMES = ["Object Segmentation"]


@pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
def test_something_with_viewer(widget_name, make_napari_viewer, napari_plugin_manager):
    napari_plugin_manager.register(napari_accelerated_pixel_and_object_classification, name=MY_PLUGIN_NAME)
    viewer = make_napari_viewer()
    num_dw = len(viewer.window._dock_widgets)
    viewer.window.add_plugin_dock_widget(
        plugin_name=MY_PLUGIN_NAME, widget_name=widget_name
    )
    assert len(viewer.window._dock_widgets) == num_dw + 1

def test_pixel_training_and_prediction(make_napari_viewer):
    viewer = make_napari_viewer()

    from napari_accelerated_pixel_and_object_classification._dock_widget import ObjectSegmentation, \
        SemanticSegmentation, ProbabilityMapping

    for Klass in [ObjectSegmentation, SemanticSegmentation, ProbabilityMapping]:

        segmenter = Klass(viewer)

        viewer.window.add_dock_widget(segmenter)

        image = np.asarray([[0,1], [2,0]])
        labels = np.asarray([[0,1], [2,0]]).astype(int)

        classifier_filename = "object_segmenter.cl"

        viewer.add_image(image)
        viewer.add_labels(labels)

        segmenter.timer.stop()
        del segmenter.timer

        segmenter.train(
            [image],
            labels,
            2,
            segmenter.feature_selector.getFeatures(),
            2,
            10,
            classifier_filename,
            False
        )

        segmenter.predict(
                    [image],
                    classifier_filename
                )

        segmenter.update_memory_consumption()
        segmenter.get_selected_annotation()
        segmenter.get_selected_annotation_data()
        segmenter.get_selected_images()
        segmenter.get_selected_images_data()

        segmenter.label_list.setCurrentIndex(0)
        segmenter.image_list.item(0).setSelected(True)
        segmenter.get_selected_annotation()
        segmenter.get_selected_annotation_data()
        segmenter.get_selected_images()
        segmenter.get_selected_images_data()

        segmenter.check_image_sizes()

def test_object_training_and_prediction(make_napari_viewer):
    viewer = make_napari_viewer()

    from napari_accelerated_pixel_and_object_classification._function import Train_object_classifier

    classifier = Train_object_classifier()

    viewer.window.add_dock_widget(classifier)

def test_feature_selector(make_napari_viewer):
    from napari_accelerated_pixel_and_object_classification._dock_widget import FeatureSelector

    viewer = make_napari_viewer()

    f = FeatureSelector(viewer.window.qt_viewer, "original gaussian_blur=1")
    f._remove_feature("original")
    f._add_feature("original")

    cb = f._make_checkbox("a", "b", True)
    cb.setChecked(False)

if __name__ == "__main__":
    import napari
    test_object_training_and_prediction(napari.Viewer)
    test_feature_selector()
    test_pixel_training_and_prediction(napari.Viewer)

def test_custom_object_classifier(make_napari_viewer):
    viewer = make_napari_viewer()

    from napari_accelerated_pixel_and_object_classification._custom_table_row_classifier import CustomObjectClassifierWidget

    classifier = CustomObjectClassifierWidget(viewer)

    viewer.window.add_dock_widget(classifier)
    