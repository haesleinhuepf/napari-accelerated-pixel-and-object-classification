# napari-accelerated-pixel-and-object-classification (APOC)

[![License](https://img.shields.io/pypi/l/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://pypi.org/project/napari-accelerated-pixel-and-object-classification)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-accelerated-pixel-and-object-classification/branch/main/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-accelerated-pixel-and-object-classification)
[![Development Status](https://img.shields.io/pypi/status/napari-accelerated-pixel-and-object-classification.svg)](https://en.wikipedia.org/wiki/Software_release_life_cycle#Alpha)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-accelerated-pixel-and-object-classification)](https://napari-hub.org/plugins/napari-accelerated-pixel-and-object-classification)
[![DOI](https://zenodo.org/badge/412525441.svg)](https://zenodo.org/badge/latestdoi/412525441)

[clesperanto](https://github.com/clEsperanto/pyclesperanto_prototype) meets [scikit-learn](https://scikit-learn.org/stable/) to classify pixels and objects in images, on a [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit) using [OpenCL](https://www.khronos.org/opencl/) in [napari].

![](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/screencast.gif)
The processed example image was kindly acquired by Daniela Vorkel, Myers lab, MPI-CBG / CSBD ([Download full video](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/demo_lund.mp4))

For using the accelerated pixel and object classifiers in python, check out [apoc](https://github.com/haesleinhuepf/apoc).
Training classifiers from pairs of image and label-mask folders is explained in 
[this notebook](https://github.com/haesleinhuepf/apoc/blob/main/demo/train_on_folders.ipynb).
For executing APOC classifiers in [Fiji](https://fiji.sc) using [clij2](https://clij.github.io) please read the documentation of the [corresponding Fiji plugin](https://github.com/clij/clijx-accelerated-pixel-and-object-classification).

![](https://github.com/clij/clijx-accelerated-pixel-and-object-classification/raw/main/docs/screenshot.png)



## Usage

### Object and Semantic Segmentation

Starting point is napari with at least one image layer and one labels layer (your annotation).

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/object_segmentation_starting_point.png)

You find Object and Semantic Segmentation in the `Tools > Segmentation / labeling`. When starting those, the following graphical user interface will show up.

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/object_and_semantic_segmentation.png)

1. Choose one or multiple images to train on. These images will be considered as multiple channels. Thus, they need to be spatially correlated. 
   Training from multiple images showing different scenes is not (yet) supported from the graphical user interface. Check out [this notebook](https://github.com/haesleinhuepf/apoc/blob/main/demo/demp_pixel_classifier_continue_training.ipynb) if you want to train from multiple image-annotation pairs.
2. Select a file where the classifier should be saved. If the file exists already, it will be overwritten.
3. Select the ground-truth annotation labels layer. 
4. Select which label corresponds to foreground (not available in Semantic Segmentation)
5. Select the feature images that should be considered for segmentation. If segmentation appears pixelated, try increasing the selected sigma values and untick `Consider original image`.
6. Tree depth and number of trees allow you to fine-tune how to deal with manifold regions of different characteristics. The higher these numbers, the longer segmentation will take. In case you use many images and many features, high depth and number of trees might be necessary. (See also `max_depth` and `n_estimators` in the [scikit-learn documentation of the Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
7. The estimation of memory consumption allows you to tune the configuration to your GPU-hardware. Also consider the GPU-hardware of others who want to use your classifier.
8. Click on Run when you're done with configuring. If the segmentation doesn't fit after the first execution, consider fine-tuning the ground-truth annotation and try again.

A successful segmentation can for example look like this:

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/object_segmentation_result.png)

After your classifier has been trained successfully, click on the "Application / Prediction" tab. If you apply the classifier again, python code will be generated. 
You can use this code for example to apply the same classifier to a folder of images. If you're new to this, check out [this notebook](https://github.com/BiAPoL/Bio-image_Analysis_with_Python/blob/main/image_processing/12_process_folders.ipynb).

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/code_generation.png)

A pre-trained classifier can be [applied from scripts as shown in the example notebook](https://github.com/haesleinhuepf/apoc/blob/main/demo/demo_object_segmenter.ipynb) or from the `Tools > Segmentation / labeling > Object segmentation (apply pretrained, APOC)`.

### Integration with the napari-assistant

Pre-trained models can also be assembled to workflows using the [napari-assistant](https://www.napari-hub.org/plugins/napari-assistant). You find APOC-operations in the categories `Filter`, `Label` and `Label Filters`:

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/assistant.png)

### Semantic segmentation

Users can also generate semantic segmentation label images where the label identifier corresponds to a class the pixel has been allocated to. 
The tool can be found in the menu `Tools > Segmentation / labeling > Semantic segmentation (APOC)`.
It works analogously like the Object Segmenter, just without the need to specify the class identifier that objects correspond to.

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/semantic_segmentation.png)

### Probability maps

The tool for generating probability maps (`Tools > Filtering > Probability Mapper (APOC)` menu) works analogously to the Object Segmenter as well. 
The only difference is that the result image is not a label image but an intensity image where the intensity represents the probability (between 0 and 1)
that a pixel belongs to a given class. In this example: The raw image (grey) has been annotated with three classes: background (black, label 1), foreground (white, label 2) and edges (grey, label 3).
The probability mapper was configured to create probability image (shown in green) for edges (label 3):

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/probability_mapper.png)

### Classifier statistics

While training, you can also activate the `Show classifier statistics` checkbox. 
When doing so, it is recommended to increase the number of trees so that the measurements are more reliable, especially when selecting many features.
This will open a small table after training where you can see how large the share of decision trees are for each analysed feature image.

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/classifier_statistics.png)

It is recommended to turn on/off the features that hold a very large share (green) or a very small share (magenta) of trees in the random forest. 
Retrain the classifier to see how the features influence the decision making.

Note: Multiple of these parameters may be correlated. 
If you select 11 feature images, which all allow to make the pixel classification similarly, but 10 of those are correlated, these 10 may appear with a share of about 0.05 while the 11th parameter has a share of 0.5. 
Thus, study these values with care.

### Merging objects

After segmentation, you can merge labeled objects using the `Tools > Segmentation post-processing > Merge objects (APOC)` menu. 
Annotate label edges that should be merged with intensity 1 and those which should be kept with intensity 2 in a blank label image.
Select which features should be considered for merging:
* `touch_portion`: The relative amount an object touches another. E.g. in a symmetric, honey-comb like tissue, neighboring cells have a touch-portion of `1/6` to each other.
* `touch_count`: The number of pixels where object touch. When using this parameter, make sure that images used for training and prediction have the same voxel size.
* `mean_touch_intensity`: The mean average intensity between touching objects. When using this parameter, make sure images used for training and prediction are normalized the same way.
* `centroid_distance`: The distance (in pixels or voxels) between centroids of labeled objects. 
* `mean_intensity_difference`: The absolute difference between the mean intensity of the two objects. This measurement allows differentiating bright and dark object and [not] mergin them.
* `standard_deviation_intensity_difference`: The absolute difference between the standard deviation of the two objects. This measurement allows to differentiate [in]homogeneous objects and [not] merge them.
* `area_difference`: The difference in area/volume/pixel-count allows differentiating small and large objects and [not] merging them.
* `mean_max_distance_to_centroid_ratio_difference`: This parameter is a shape descriptor, similar to elongation, allowing to differentiate roundish and elongate object and [not] merging them.

Note: most features are recommended to be used in isotropic images only.

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/merge_objects1.png)

For training, use an image with equivalized intensity (1), an over-segmented label image (2) and annotations (3). When drawing annotations in a new labels layer, make sure to misguide the algorithm draw on edges of touching objects a 1 if those should be merged and a 2 if they should be kept. Make sure there are no 1/2 annotation circles on both: labels which should be merged and kept.

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/merge_objects2.png)

### Object classification

Click the menu `Tools > Segmentation post-processing > Object classification (APOC)`. 

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/menu.png)

This user interface will be shown:

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/object_classifier_gui.png)

1. The image layer will be used for intensity based feature extraction (see below).
2. The labels layer should be contain the segmentation of objects that should be classified. 
   You can use the Object Segmenter explained above to create this layer.
3. The annotation layer should contain manual annotations of object classes. 
   You can draw lines crossing single and multiple objects of the same kind. 
   For example draw a line through some elongated objects with label "1" and another line through some rather roundish objects with label "2".
   If these lines touch the background, that will be ignored.
4. Tree depth and number of trees allow you to fine-tune how to deal with manifold objects of different characteristics. The higher these numbers, the longer classification will take. In case you use many features, high depth and number of trees might be necessary. (See also `max_depth` and `n_estimators` in the [scikit-learn documentation of the Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
5. Select the right features for training. For example, for differentiating objects according to their shape as suggested above, select "shape".
   The features are extracted using clEsperanto and are shown by example in [this notebook](https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/tissues/parametric_maps.ipynb).
6. Click on the `Run` button. If classification doesn't perform well in the first attempt, try changing selected features.  

If classification worked well, it may for example look like this. Note the two thick lines which were drawn to annotate elongated and roundish objects with brown and cyan:

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/object_classification_result.png)

A pre-trained model can later be applied [from scripts as shown in the example notebook](https://github.com/haesleinhuepf/apoc/blob/main/demo/cell_classification.ipynb) or using the menu `Tools > Segmentation post-processing > Object classification (apply pretrained, APOC)`.

### Feature correlation matrix

When training object classifiers it is crucial to investigate to which degree features are correlated and select the right, ideally uncorrelated features to classify objects robustly.
After measuring features with any compatible napari plugin listed below, you can visualize the feature correlation matrix using the menu `Tools > Measurement > Show feature correlation matrix (pandas, APOC)` and by selecting the labels layer which has been analyzed.
Before computing the correlation matrix, all rows containing [NaN](https://en.wikipedia.org/wiki/NaN) values are removed.
For further details, please refer to the [documentation of the underlying function in pandas](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html).

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/feature_correlation_matrix.png)

### Surface Vertex Classification (SVeC)

When using napari-APOC in combination with [napari-process-points-and-surfaces>=0.3.3](https://github.com/haesleinhuepf/napari-process-points-and-surfaces), 
one can also classify vertices. Therefore, use for example the menu `Measurement > Surface quality table (vedo, nppas)` to determine quantitative measurements
and the menu `Surfaces > Annotate surface manually (nppas)` for manual annotations. It is recommended to annotate the entire surface with value 1 as background, and specific regions of interest with integer numbers > 1.
After measurements have been extracted and annotations were made, start SVeC from the `Surfaces > Surface vertex classification (custom properties, APOC)` menu. It can be used like the Object Classifier explained above.

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/demo_vertex_classification.gif)

[Download full video](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/demo_vertex_classification.mp4)

### Classifier statistics
After classifier training, you can study the share of the individual features/measurements and how they are correlated by activating the checkboxes `Show classifier statistics` and `Show feature correlation matrix`.

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/correlation_matrix2.png)

This can help understanding how the classifier works. Furthermore, you can accelerate the classifier by reducing the number of correlated features.

### Object classification from custom measurements

You can also classify labeled objects according to custom measurements. For deriving those measurements, you can use these napari plugins:

* [morphometrics](https://www.napari-hub.org/plugins/morphometrics)
* [PartSeg](https://www.napari-hub.org/plugins/PartSeg)
* [napari-simpleitk-image-processing](https://www.napari-hub.org/plugins/napari-simpleitk-image-processing)
* [napari-cupy-image-processing](https://www.napari-hub.org/plugins/napari-cupy-image-processing)
* [napari-pyclesperanto-assistant](https://www.napari-hub.org/plugins/napari-pyclesperanto-assistant)
* [napari-skimage-regionprops](https://www.napari-hub.org/plugins/napari-skimage-regionprops)

Furthermore, if you use napari from Python, you can also create a dictionary or pandas DataFrame with measurements and store it in the `labels_layer.features` to make them available in the object classifier.

After labels have been measured, you can start the `Object Classifier (custom properties, APOC)` from the `Tools > Segmentation post-processing` menu:

![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/table_row_classifier_gui.png)

1. Select the labels layers that has been measured.
2. The annotation layer should contain manual annotations of object classes. 
   You can draw lines crossing single and multiple objects of the same kind. 
   For example draw a line through some elongated objects with label "1" and another line through some rather roundish objects with label "2".
   If these lines touch the background, that will be ignored.
3. Select the measurements / features that should be used for object classification.
4. Use the `Update Measurements` button in case you did new measurements after Object classifier dialog was opened.
5. Enter the filename of the classifier to be trained here. This file will be overwritten in case it existed already.
6. Tree depth and number of trees allow you to fine-tune how to deal with manifold objects of different characteristics. The higher these numbers, the longer classification will take. In case you use many features, high depth and number of trees might be necessary. (See also `max_depth` and `n_estimators` in the [scikit-learn documentation of the Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
7. The classification result will be stored under this name in the labels-layer's properties.
8. Choose if the results table should be shown. Choose if classifier statistics should be shown. [Read more about classifier statistics](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/27_cell_classification/forest_statistics.html).
9. Click on `Run` to start training and prediction.

You can also train those classifiers from Python and reuse them: [Read more about using the TableRowClassifier from python](https://haesleinhuepf.github.io/BioImageAnalysisNotebooks/27_cell_classification/apoc_simpleitk_object_classification.html)

### Classifier statistics and correlation matrix
After classifier training, you can study the share of the individual features/measurements and how they are correlated by activating the checkboxes `Show classifier statistics` and `Show correlation matrix`.
![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/correlation_matrix.png)

This can help understanding how the classifier works. Furthermore, you can accelerate the classifier by reducing the number of correlated features.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

It is recommended to install the plugin in a conda environment. Therefore install conda first, e.g. [mini-conda](https://docs.conda.io/en/latest/miniconda.html).
If you never worked with conda before, reading this [short introduction](https://github.com/BiAPoL/Bio-image_Analysis_with_Python/blob/main/conda_basics/01_conda_environments.md) might be helpful.

Optional: Setup a fresh conda environment, activate it and install napari:

```
conda create --name napari_apoc python=3.9
conda activate napari_apoc
conda install napari
```

If your conda environment is set up, you can install `napari-accelerated-pixel-and-object-classification` using [pip]. Note: you need [pyopencl](https://documen.tician.de/pyopencl/) first.

```
conda install -c conda-forge pyopencl
pip install napari-accelerated-pixel-and-object-classification
```

Mac-users please also install this:

    conda install -c conda-forge ocl_icd_wrapper_apple
    
Linux users please also install this:
    
    conda install -c conda-forge ocl-icd-system


## Contributing
 
Contributions, feedback and suggestions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## Similar napari plugins
There are other plugins with similar functionality for interactive classification of pixels and objects.

* [napari-feature-classifier](https://github.com/fractal-napari-plugins-collection/napari-feature-classifier)

## License

Distributed under the terms of the [BSD-3] license,
"napari-accelerated-pixel-and-object-classification" is free and open source software

## Issues

If you encounter any problems, please [open a thread on image.sc](https://image.sc) along with a detailed description and tag [@haesleinhuepf](https://github.com/haesleinhuepf).

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
[file an issue]: https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/issues
[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
