def wrap_api(func):
    func.__module__ = "napari_accelerated_pixel_and_object_classification"
    return func
