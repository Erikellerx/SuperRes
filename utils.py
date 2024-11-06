
from baseline.SRCNN import SRCNN
from baseline.FSRCNN import FSRCNN

def build_model(model_name):
    if model_name == 'SRCNN':
        print("======>  Baseline Model: SRCNN  <======")
        return SRCNN()
    elif model_name == 'FSRCNN':
        print("======>  Baseline Model: FSRCNN  <======")
        return FSRCNN()
    else:
        raise ValueError("Model not found")