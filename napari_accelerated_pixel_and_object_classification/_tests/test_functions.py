import numpy as np

def test_training(make_napari_viewer):
    viewer = make_napari_viewer()

    from napari_accelerated_pixel_and_object_classification._function import Train_object_classifier,\
        Train_pixel_classifier,\
        Train_object_segmentation,\
        Train_object_segmentation_from_visible_image_layers,\
        Train_pixel_classifier_from_visible_image_layers,\
        Connected_component_labeling,\
        Apply_pixel_classification,\
        Apply_object_segmentation,\
        Apply_probability_mapper,\
        Apply_object_segmentation_to_visible_image_layers,\
        Apply_pixel_classification_to_visible_image_layers
        # Apply_object_classification,\

    import apoc

    image = np.asarray([
        [0,0,1,1],
        [0,0,1,1],
        [2,2,1,1],
        [2,2,1,1],
    ])
    labels = image.astype(int)

    viewer.add_image(image)
    viewer.add_labels(labels)

    #Train_object_classifier()(image, labels, labels)
    Train_object_classifier()(image, labels, labels, "ObjectClassifier.cl", 2, 10, True, True, True, True, True, True, True, True, True, True, True, True, True)
    Train_pixel_classifier(image, labels, featureset=apoc.PredefinedFeatureSet.custom, custom_features="original")
    Train_pixel_classifier(image, labels)
    #Train_object_segmentation(image, labels)
    Train_object_segmentation_from_visible_image_layers(labels, napari_viewer=viewer, featureset=apoc.PredefinedFeatureSet.custom, custom_features="original")
    Train_object_segmentation_from_visible_image_layers(labels, napari_viewer=viewer)
    Train_pixel_classifier_from_visible_image_layers(labels, napari_viewer=viewer, featureset=apoc.PredefinedFeatureSet.custom, custom_features="original")
    Train_pixel_classifier_from_visible_image_layers(labels, napari_viewer=viewer)
    Connected_component_labeling(labels)
    Connected_component_labeling(labels, fill_gaps_between_labels=False)
    #Apply_object_classification(image, labels)
    Apply_pixel_classification(image)
    #Apply_object_segmentation(image)
    #Apply_probability_mapper(image)
    Apply_object_segmentation_to_visible_image_layers(napari_viewer=viewer)
    Apply_pixel_classification_to_visible_image_layers(napari_viewer=viewer)

def test_object_segmentation():

    from napari_accelerated_pixel_and_object_classification._function import Train_object_segmentation,\
        Apply_object_segmentation
    import pyclesperanto_prototype as cle

    image = cle.push(np.asarray([
        [0,1],
        [2,0]]))
    labels = cle.push(np.asarray([
        [1,2],
        [2,1]]).astype(int))

    import apoc
    Train_object_segmentation(image, labels, model_filename="file.cl", featureset=apoc.PredefinedFeatureSet.custom, custom_features="original")
    Apply_object_segmentation(image, model_filename="file.cl")

def test_probability_mapper():

    from napari_accelerated_pixel_and_object_classification._function import Train_probability_mapper, \
        Apply_probability_mapper
    import pyclesperanto_prototype as cle

    image = cle.push(np.asarray([
        [0,1],
        [2,0]]))
    labels = cle.push(np.asarray([
        [1,2],
        [2,1]]).astype(int))

    import apoc
    Train_probability_mapper(image, labels, model_filename="file2.cl", featureset=apoc.PredefinedFeatureSet.custom, custom_features="original")
    Apply_probability_mapper(image, model_filename="file2.cl")
