import logging

# Reference: https://github.com/lehaifeng/T-GCN/tree/master/T-GCN/T-GCN-PyTorch


def format_logger(logger, fmt="\033[31m[%(asctime)s %(levelname)s]\033[0m%(message)s"):
    """Formats the logger messange"""
    handler = logger.handlers[0]
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)


def output_logger_to_file(
    logger, output_path, fmt="[%(asctime)s %(levelname)s]%(message)s"
):
    """Save the output logger message to a file path"""
    handler = logging.FileHandler(output_path, encoding="UTF-8")
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
