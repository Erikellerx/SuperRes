
from baseline.SRCNN import SRCNN

def get_model(model_name):
    if model_name == 'SRCNN':
        print("======>  Baseline Model: SRCNN  <======")
        return SRCNN()
    else:
        raise ValueError("Model not found")