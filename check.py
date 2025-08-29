import pickle, os
from sklearn.utils.validation import check_is_fitted

with open("model-2.pkl","rb") as f:
    m = pickle.load(f)

print("Loaded object:", type(m))
# This will raise NotFittedError if it’s unfitted
check_is_fitted(m)
