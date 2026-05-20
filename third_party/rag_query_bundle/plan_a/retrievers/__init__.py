__all__ = ["make_scoped_retriever"]


def make_scoped_retriever(*args, **kwargs):
    from plan_a.retrievers.scoped import make_scoped_retriever as _impl

    return _impl(*args, **kwargs)
