from pylsl import StreamInlet, resolve_byprop

# Modify this to match your device's stream name!
device_name = "Explore_8441_ExG" #Explore_XXXX_ExG

def get_stream_inlet(name=device_name, timeout=5.0):
    """
    Returns a StreamInlet for the specified stream name.
    """
    streams = resolve_byprop('name', name, timeout=timeout)
    if not streams:
        raise RuntimeError(f"{name} stream not found")
    inlet = StreamInlet(streams[0])
    return inlet

def has_lsl_stream(name: str = device_name, timeout: float = 1.0) -> bool:
    """Return True if a stream with the given name is found within timeout."""
    try:
        streams = resolve_byprop('name', name, timeout=timeout)
        return bool(streams)
    except Exception:
        return False