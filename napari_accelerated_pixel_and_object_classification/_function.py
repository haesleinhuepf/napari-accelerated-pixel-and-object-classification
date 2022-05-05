from apoc import PredefinedFeatureSet, PixelClassifier, ObjectSegmenter, ObjectClassifier, ProbabilityMapper

import numpy as np
from napari_plugin_engine import napari_hook_implementation

import napari
from napari_time_slicer import time_slicer
from napari_tools_menu import register_function, register_dock_widget
from magicgui import magic_factory

from qtpy.QtWidgets import QTableWidget

@napari_hook_implementation
def napari_experimental_provide_function():
    return [
        Train_object_segmentation,
        Apply_object_segmentation,
        Train_probability_mapper,
        Apply_probability_mapper,
        Train_object_segmentation_from_visible_image_layers,
        Apply_object_segmentation_to_visible_image_layers,
        Train_pixel_classifier,
        Apply_pixel_classification,
        Train_pixel_classifier_from_visible_image_layers,
        Apply_pixel_classification_to_visible_image_layers,
        Connected_component_labeling, Apply_object_classification]

def Train_pixel_classifier(
        image: "napari.types.ImageData",
        annotation : "napari.types.LabelsData",
        model_filename : str = "PixelClassifier.cl",
        featureset : PredefinedFeatureSet = PredefinedFeatureSet.small_quick,
        custom_features : str = "original gaussian_blur=1 sobel_of_gaussian_blur=1",
        max_depth : int = 2,
        num_ensembles : int = 10
) -> "napari.types.LabelsData":
    feature_stack = featureset.value
    if feature_stack == "":
        feature_stack = custom_features

    clf = PixelClassifier(opencl_filename=model_filename, num_ensembles=num_ensembles, max_depth=max_depth)
    clf.train(feature_stack, annotation, image)

    result = clf.predict(features=feature_stack, image=image)
    return result

def Train_probability_mapper(
        image: "napari.types.ImageData",
        annotation : "napari.types.LabelsData",
        model_filename : str = "ProbabilityMapper.cl",
        featureset : PredefinedFeatureSet = PredefinedFeatureSet.small_quick,
        custom_features : str = "original gaussian_blur=1 sobel_of_gaussian_blur=1",
        output_probability_of_class : int = 2,
        max_depth : int = 2,
        num_ensembles : int = 10
) -> "napari.types.LabelsData":
    feature_stack = featureset.value
    if feature_stack == "":
        feature_stack = custom_features

    clf = ProbabilityMapper(opencl_filename=model_filename, num_ensembles=num_ensembles, max_depth=max_depth, output_probability_of_class=output_probability_of_class)
    clf.train(feature_stack, annotation, image)

    result = clf.predict(features=feature_stack, image=image)
    return result

@register_function(menu="Segmentation / labeling > Semantic segmentation (apply pretrained, APOC)")
@time_slicer
def Apply_pixel_classification(image: "napari.types.ImageData",
                               model_filename : str = "PixelClassifier.cl",
                               viewer: napari.Viewer = None) -> "napari.types.LabelsData":

    clf = PixelClassifier(opencl_filename=model_filename)
    print("Hello world")
    result = clf.predict(image=[image])
    print("Result is ", result.shape, result.dtype)
    return result

def Train_pixel_classifier_from_visible_image_layers(
        annotation : "napari.types.LabelsData",
        model_filename : str = "PixelClassifier.cl",
        featureset : PredefinedFeatureSet = PredefinedFeatureSet.small_quick,
        custom_features : str = "original gaussian_blur=1 sobel_of_gaussian_blur=1",
        max_depth : int = 2,
        num_ensembles : int = 10,
        napari_viewer : napari.Viewer = None
) -> "napari.types.LabelsData":
    image = [layer.data for layer in napari_viewer.layers if (isinstance(layer, napari.layers.Image) and layer.visible)]

    feature_stack = featureset.value
    if feature_stack == "":
        feature_stack = custom_features

    clf = PixelClassifier(opencl_filename=model_filename, num_ensembles=num_ensembles, max_depth=max_depth)
    clf.train(feature_stack, annotation, image)

    result = clf.predict(features=feature_stack, image=image)
    return result

def Apply_pixel_classification_to_visible_image_layers(
        model_filename : str = "PixelClassifier.cl",
        napari_viewer : napari.Viewer = None
) -> "napari.types.LabelsData":
    image = [layer.data for layer in napari_viewer.layers if (isinstance(layer, napari.layers.Image) and layer.visible)]

    clf = PixelClassifier(opencl_filename=model_filename)
    result = clf.predict(image=image)
    return result

def Train_object_segmentation(
        image: "napari.types.ImageData",
        annotation : "napari.types.LabelsData",
        model_filename : str = "ObjectSegmenter.cl",
        featureset : PredefinedFeatureSet = PredefinedFeatureSet.small_quick,
        custom_features : str = "original gaussian_blur=1 sobel_of_gaussian_blur=1",
        max_depth : int = 2,
        num_ensembles : int = 10,
        annotated_object_intensity : int = 2
) -> "napari.types.LabelsData":
    feature_stack = featureset.value
    if feature_stack == "":
        feature_stack = custom_features

    clf = ObjectSegmenter(opencl_filename=model_filename, num_ensembles=num_ensembles, max_depth=max_depth, positive_class_identifier=annotated_object_intensity)
    clf.train(feature_stack, annotation, [image])

    result = clf.predict(feature_stack, [image])
    return result


@register_function(menu="Filtering > Probability Mapper (apply pretrained, APOC)")
@time_slicer
def Apply_probability_mapper(image: "napari.types.ImageData",
                              model_filename : str = "ProbabilityMapper.cl",
                              viewer: napari.Viewer = None) -> "napari.types.ImageData":
    clf = ProbabilityMapper(opencl_filename=model_filename)
    result = clf.predict(image=image)
    return result


@register_function(menu="Segmentation / labeling > Object segmentation (apply pretrained, APOC)")
@time_slicer
def Apply_object_segmentation(image: "napari.types.ImageData",
                              model_filename : str = "ObjectSegmenter.cl",
                              viewer: napari.Viewer = None) -> "napari.types.LabelsData":
    clf = ObjectSegmenter(opencl_filename=model_filename)
    result = clf.predict(image=[image])
    return result

def Train_object_segmentation_from_visible_image_layers(
        annotation : "napari.types.LabelsData",
        model_filename : str = "ObjectSegmenter.cl",
        featureset : PredefinedFeatureSet = PredefinedFeatureSet.small_quick,
        custom_features : str = "original gaussian_blur=1 sobel_of_gaussian_blur=1",
        max_depth : int = 2,
        num_ensembles : int = 10,
        annotated_object_intensity : int = 2,
        napari_viewer : napari.Viewer = None
) -> "napari.types.LabelsData":
    image = [layer.data for layer in napari_viewer.layers if (isinstance(layer, napari.layers.Image) and layer.visible)]

    feature_stack = featureset.value
    if feature_stack == "":
        feature_stack = custom_features

    clf = ObjectSegmenter(opencl_filename=model_filename, num_ensembles=num_ensembles, max_depth=max_depth, positive_class_identifier=annotated_object_intensity)
    clf.train(feature_stack, annotation, image)

    result = clf.predict(features=feature_stack, image=image)
    return result

def Apply_object_segmentation_to_visible_image_layers(
        model_filename : str = "ObjectSegmenter.cl",
        napari_viewer : napari.Viewer = None
) -> "napari.types.LabelsData":
    image = [layer.data for layer in napari_viewer.layers if (isinstance(layer, napari.layers.Image) and layer.visible)]

    clf = ObjectSegmenter(opencl_filename=model_filename)
    result = clf.predict(image=image)
    return result


def Connected_component_labeling(labels: "napari.types.LabelsData", object_class_identifier : int = 2, fill_gaps_between_labels:bool = True) -> "napari.types.LabelsData":
    import pyclesperanto_prototype as cle
    binary = cle.equal_constant(labels, constant=object_class_identifier)
    if fill_gaps_between_labels:
        instances = cle.voronoi_labeling(binary)
    else:
        instances = cle.connected_components_labeling_box(binary)
    return instances

@register_dock_widget(menu="Segmentation post-processing > Object classification (APOC)")
@magic_factory(
    model_filename=dict(widget_type='FileEdit', mode='w'),
    shape_extension_ratio=dict(label='shape (extension ratio)')
)
def Train_object_classifier(image: "napari.types.ImageData",
                            labels : "napari.types.LabelsData",
                            annotation : "napari.types.LabelsData",
                            model_filename : "magicgui.types.PathLike" = "ObjectClassifier.cl",
                            max_depth : int = 2,
                            num_ensembles : int = 10,
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
                            show_classifier_statistics=False,
                            show_feature_correlation_matrix=False,
                            viewer : "napari.Viewer" = None
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

    # remove first and last comma
    features = features[1:-1]

    clf = ObjectClassifier(opencl_filename=model_filename, num_ensembles=num_ensembles,
                                           max_depth=max_depth)

    clf.train(features, labels, annotation, image)
    result = clf.predict(labels, image)

    if viewer is not None:
        from napari_workflows._workflow import _get_layer_from_data
        layer = _get_layer_from_data(viewer, labels)
        if layer is not None:
            for key, item in clf._data.items():
                print(key, type(item), item.shape)
            layer.properties = clf._data

            if show_feature_correlation_matrix:
                show_feature_correlation_matrix(layer, viewer)

        if show_classifier_statistics:
            from ._dock_widget import update_model_analysis
            table = QTableWidget()
            update_model_analysis(table, clf)
            viewer.window.add_dock_widget(table, name="Classifier statistics")


    return result


@register_function(menu="Measurement > Feature correlation matrix (pandas, APOC)")
def show_feature_correlation_matrix(layer: "napari.layers.Layer", viewer:napari.Viewer = None):
    from ._dock_widget import update_table_gui
    import pandas as pd
    correlation_matrix = pd.DataFrame(layer.properties).dropna().corr()

    if viewer is not None:
        table = QTableWidget()
        table.setColumnCount(len(correlation_matrix))
        table.setRowCount(len(correlation_matrix))

        update_table_gui(table, correlation_matrix, minimum_value=-1, maximum_value=1)
        viewer.window.add_dock_widget(table, name="Feature correlation matrix")
    else:
        return correlation_matrix


@register_function(menu="Segmentation post-processing > Object classification (apply pretrained, APOC)")
@time_slicer
def Apply_object_classification(image: "napari.types.ImageData",
                             labels: "napari.types.LabelsData",
                             model_filename : str = "ObjectClassifier.cl",
                             viewer: napari.Viewer = None) -> "napari.types.LabelsData":

    clf = ObjectClassifier(opencl_filename=model_filename)
    result = clf.predict(labels, image)
    return result
