__all__ = ["make_rag_tool"]


def make_rag_tool(*args, **kwargs):
    from plan_a.tools.rag import make_rag_tool as _impl

    return _impl(*args, **kwargs)
