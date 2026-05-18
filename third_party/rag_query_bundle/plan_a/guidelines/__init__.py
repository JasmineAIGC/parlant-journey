__all__ = ["setup_guidelines"]


def setup_guidelines(*args, **kwargs):
    from plan_a.guidelines.setup import setup_guidelines as _impl

    return _impl(*args, **kwargs)
