"""
Device configuration for ML models - M2 MPS acceleration support.
"""
import os
import logging

# Suppress TensorFlow warnings before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger = logging.getLogger(__name__)

_DEVICE_CONFIGURED = False
_CURRENT_DEVICE = None


def configure_tensorflow_device() -> str:
    """
    Configure TensorFlow to use M2 MPS acceleration.

    On Apple Silicon (M1/M2/M3), TensorFlow can use the Metal Performance Shaders
    (MPS) backend for GPU acceleration. This function detects and configures
    the optimal device.

    Returns:
        str: Device being used ('gpu' for MPS, 'cpu' otherwise)
    """
    global _DEVICE_CONFIGURED, _CURRENT_DEVICE

    if _DEVICE_CONFIGURED:
        return _CURRENT_DEVICE

    import tensorflow as tf

    # List all physical devices
    physical_devices = tf.config.list_physical_devices()
    logger.debug(f"Available devices: {physical_devices}")

    # Check for GPU (MPS on Apple Silicon)
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Enable memory growth to prevent OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            logger.info(f"MPS GPU acceleration enabled: {gpus}")
            _CURRENT_DEVICE = 'gpu'

        except RuntimeError as e:
            logger.warning(f"Could not configure GPU memory growth: {e}")
            _CURRENT_DEVICE = 'cpu'
    else:
        logger.info("No GPU found, using CPU")
        _CURRENT_DEVICE = 'cpu'

    _DEVICE_CONFIGURED = True
    return _CURRENT_DEVICE


def get_device_info() -> dict:
    """
    Get detailed information about available devices.

    Returns:
        dict: Device information including type, name, and memory info
    """
    import tensorflow as tf

    info = {
        'tensorflow_version': tf.__version__,
        'devices': [],
        'gpu_available': False,
        'mps_available': False
    }

    physical_devices = tf.config.list_physical_devices()

    for device in physical_devices:
        device_info = {
            'name': device.name,
            'type': device.device_type
        }
        info['devices'].append(device_info)

        if device.device_type == 'GPU':
            info['gpu_available'] = True
            # On Apple Silicon, GPU means MPS is available
            info['mps_available'] = True

    return info


def log_device_info():
    """Log device information for debugging."""
    info = get_device_info()

    logger.info("=" * 50)
    logger.info("TensorFlow Device Configuration")
    logger.info("=" * 50)
    logger.info(f"TensorFlow Version: {info['tensorflow_version']}")
    logger.info(f"GPU Available: {info['gpu_available']}")
    logger.info(f"MPS Available: {info['mps_available']}")

    for device in info['devices']:
        logger.info(f"  Device: {device['name']} ({device['type']})")

    logger.info("=" * 50)
