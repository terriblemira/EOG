from pylsl import StreamInlet, resolve_byprop

def get_stream_inlet(name='Explore_8441_ExG', timeout=5.0):
    """
    Returns a StreamInlet for the specified stream name.
    """
    streams = resolve_byprop('name', name, timeout=timeout)
    if not streams:
        raise RuntimeError(f"{name} stream not found")
    inlet = StreamInlet(streams[0])
    return inlet