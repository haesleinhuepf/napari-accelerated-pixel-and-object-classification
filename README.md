# napari-accelerated-pixel-and-object-classification (APOC)

[![License](https://img.shields.io/pypi/l/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://pypi.org/project/napari-accelerated-pixel-and-object-classification)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-accelerated-pixel-and-object-classification.svg?color=green)](https://python.org)
[![tests](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/workflows/tests/badge.svg)](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/actions)
[![codecov](https://codecov.io/gh/haesleinhuepf/napari-accelerated-pixel-and-object-classification/branch/main/graph/badge.svg)](https://codecov.io/gh/haesleinhuepf/napari-accelerated-pixel-and-object-classification)

[clEsperanto](https://github.com/clEsperanto/pyclesperanto_prototype) meets [scikit-learn](https://scikit-learn.org/stable/)

A yet experimental OpenCL-based Random Forest Classifier for pixel and labeled object classification in [napari].

![](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/screenshot.png)
The processed example image [maize_clsm.tif](https://github.com/dlegland/mathematical_morphology_with_MorphoLibJ/blob/main/sampleImages/maize_clsm.tif)
is licensed by David Legland under 
[CC-BY 4.0 license](https://github.com/dlegland/mathematical_morphology_with_MorphoLibJ/blob/main/LICENSE)

For using the accelerated pixel and object classifiers in python, check out [apoc](https://github.com/haesleinhuepf/apoc).


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

## Installation

You can install `napari-accelerated-pixel-and-object-classification` via [pip]. Note: you also need [pyopencl](https://documen.tician.de/pyopencl/).

    conda install pyopencl
    pip install napari-accelerated-pixel-and-object-classification
    
In case of issues in napari, make sure these dependencies are installed properly:
    
    pip install pyclesperanto_prototype
    pip install apoc

## Usage
[documentation work in progress]

Open an image in napari and add a labels layer. Annotate foreground and background with two different label identifiers. You can also add a third, e.g. a membrane-like region in between to improve segmentation quality.
![img.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/img.png)

Click the menu `Plugins > Segmentation (Accelerated Pixel and Object Classification) > Train pixel classifier`. 
Consider changing the `featureset`. There are three options for selecting 
small (about 1 pixel sized) objects, 
medium (about 5 pixel sized) object and 
large (about 25 pixel sized) objects.
Make sure the right image and annotation layers are selected and click on `Run`.

![img_1.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/img_1.png)

The classifier was saved as `temp.cl` to disc. You can later re-use it by clicking the menu `Plugins > OpenCL Random Forest Classifiers > Predict pixel classifier`

Optional: Hide the annotation layer.

Click the menu `Plugins > Segmentation (Accelerated Pixel and Object Classification) > Connected Component Labeling`.
Make sure the right labels layer is selected. It is supposed to be the result layer from the pixel classification.
Select the `object class identifier` you used for annotating objects, that's the intensity you drew on objects in the annotation layer.
Hint: If you want to analyse touching neigbors afterwards, activate the `fill gaps between labels` checkbox.
Click on the `Run` button.
![img_2.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/img_2.png)

Optional: Hide the pixel classification result layer. Change the opacity of the connected component labels layer.

Add a new labels layer and annotate different object classes by drawing lines through them. 
In the following example objects with different size and shape were annotated in three classes:
* round, small
* round, large
* elongated
![img_3.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/img_3.png)
  
Click the menu `Plugins > Segmentation (Accelerated Pixel and Object Classification) > Train object classifier`. Select the right layers for training.
The labels layer should be the result from connected components labeling.
The annotation layer should be the just annotated object classes layer.
Select the right features for training. Click on the `Run` button. 
After training, the classifier will be stored to disc in the file you specified.
You can later re-use it by clicking the menu `Plugins > Segmentation (Accelerated Pixel and Object Classification) > Predict label classifier`

![img_5.png](https://github.com/haesleinhuepf/napari-accelerated-pixel-and-object-classification/raw/main/images/img_5.png)

This is an experimental napari plugin. Feedback is very welcome!

## Contributing
 
Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

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
