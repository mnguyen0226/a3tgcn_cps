import logging

# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch


def format_logger(logger, fmt="\033[31m[%(asctime)s %(levelname)s]\033[0m%(message)s"):
    """Sets format of logging message

    Args:
        logger
        fmt (str, optional). Defaults to "\033[31m[%(asctime)s %(levelname)s]\033[0m%(message)s".
    """
    handler = logger.handlers[0]
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)


def output_logger_to_file(
    logger, output_path, fmt="[%(asctime)s %(levelname)s]%(message)s"
):
    """Adds logger to output file path

    Args:
        logger
        output_path
        fmt (str, optional). Defaults to "[%(asctime)s %(levelname)s]%(message)s".
    """
    handler = logging.FileHandler(output_path, encoding="UTF-8")
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
