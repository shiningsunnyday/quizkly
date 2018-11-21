import os
import pickle
import sys

if __name__ == "__main__":
    basepath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__),'model')

    with open(os.path.join(basepath, 'gaps.gsm'), 'rb') as model:
        gsm = pickle.load(model)
    gsm.classifier._dual_coef_ = gsm.classifier.dual_coef_
    with open(os.path.join(basepath, 'gaps2.gsm'), 'wb') as output:
        pickle.dump(gsm, output, pickle.HIGHEST_PROTOCOL)
