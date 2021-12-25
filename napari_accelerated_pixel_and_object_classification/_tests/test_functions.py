import numpy as np

def test_training(make_napari_viewer):
    viewer = make_napari_viewer()

    from napari_accelerated_pixel_and_object_classification._function import Train_object_classifier,\
        Train_pixel_classifier,\
        Train_object_segmentation,\
        Train_object_segmentation_from_visible_image_layers,\
        Train_pixel_classifier_from_visible_image_layers,\
        Connected_component_labeling,\
        Apply_object_classification,\
        Apply_pixel_classification,\
        Apply_object_segmentation,\
        Apply_probability_mapper,\
        Apply_object_segmentation_to_visible_image_layers,\
        Apply_pixel_classification_to_visible_image_layers

    image = np.asarray([[0,1], [2,0]])
    labels = np.asarray([[0,1], [2,0]]).astype(int)

    viewer.add_image(image)
    viewer.add_labels(labels)

    Train_object_classifier(image, labels, labels)
    Train_pixel_classifier(image, labels)
    #Train_object_segmentation(image, labels)
    Train_object_segmentation_from_visible_image_layers(labels, napari_viewer=viewer)
    Train_pixel_classifier_from_visible_image_layers(labels, napari_viewer=viewer)
    Connected_component_labeling(labels)
    Apply_object_classification(image, labels)
    Apply_pixel_classification(image)
    #Apply_object_segmentation(image)
    #Apply_probability_mapper(image)
    Apply_object_segmentation_to_visible_image_layers(napari_viewer=viewer)
    Apply_pixel_classification_to_visible_image_layers(napari_viewer=viewer)