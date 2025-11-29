import asyncio
import logging
import torch

def logger_config():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    MAX_CONCURRENT_INFERENCES = 2

    infer_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
    return logger, infer_semaphore

def get_device_index() -> int:
    """
    Возвращает device индекс для transformers.pipeline:
    -1 -> CPU
     0 -> первая CUDA-устройство
    """
    try:
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return -1