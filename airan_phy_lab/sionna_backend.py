def sionna_available() -> bool:
    try:
        import sionna, tensorflow as tf
        return True
    except Exception:
        return False

def require_sionna():
    if not sionna_available():
        raise ImportError('Sionna/TensorFlow not available. Install with `pip install sionna tensorflow` in a compatible CUDA environment.')

def describe_sionna_environment() -> dict:
    require_sionna(); import sionna, tensorflow as tf
    gpus=tf.config.list_physical_devices('GPU')
    return {'sionna_version':getattr(sionna,'__version__','unknown'),'tensorflow_version':tf.__version__,'num_gpus':len(gpus),'gpus':[str(g) for g in gpus]}

def run_sionna_smoke_test():
    return {'status':'ok','environment':describe_sionna_environment(),'next_step':'Implement LDPC + TDL + OFDM BLER sweep.'}
