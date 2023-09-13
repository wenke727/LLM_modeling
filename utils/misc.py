from bigmodelvis import Visualization


def vis_model_stucture(model):
    return Visualization(model).structure_graph()

def set_logger(fn, level="DEBUG"):
    import pandas as pd
    from loguru import logger
    
    pd.set_option('display.max_colwidth', 200)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 5000)
    pd.set_option('display.float_format', '{:.3f}'.format)

    with open(fn, 'w'):
        pass
    
    logger.remove()
    logger.add(fn, rotation="500 MB", level="DEBUG")
    # 也输出到控制台，但只显示INFO级别及以上的日志
    logger.add(
        sink=lambda msg: print(msg, end=""), level=level, colorize=True
    )  

    return logger