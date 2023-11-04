try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.14.1"

__common_alias__ = "napoc"

from ._function import apply_object_selection, apply_object_classification, apply_object_segmentation, \
    apply_pixel_classification, apply_probability_mapper
from ._object_merger import apply_object_merger

from ._function import napari_experimental_provide_function
from ._dock_widget import napari_experimental_provide_dock_widget
