try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.14.1"

__common_alias__ = "napoc"

import sys

if sys.platform == "darwin":
    import os
    os.environ.setdefault("CLE_NO_DOUBLE_PRECISION", "1")
    try:
        import pyclesperanto_prototype as cle

        if hasattr(cle, "enable_double_precision"):
            cle.enable_double_precision(False)
    except Exception:
        pass

from ._function import apply_object_selection, apply_object_classification, apply_object_segmentation, \
    apply_pixel_classification, apply_probability_mapper
from ._object_merger import apply_object_merger

from ._function import napari_experimental_provide_function
from ._dock_widget import napari_experimental_provide_dock_widget
