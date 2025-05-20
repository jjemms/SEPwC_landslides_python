import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import pytest
from terrain_analysis import make_classifier

def test_make_classifier_verbose(capsys):
    # make a simple separable dataset
    N = 40
    X = pd.DataFrame({
        'x1': np.concatenate([np.zeros(N//2), np.ones(N//2)]),
        'x2': np.concatenate([np.zeros(N//2), np.ones(N//2)]) * 2
    })
    y = np.array([0]*(N//2) + [1]*(N//2))

    # run this with verbose=True
    clf = make_classifier(X, y, verbose=True)

    # capture the output
    out = capsys.readouterr().out
    # check that the output contains the expected strings
    assert "Train accuracy:" in out, f"Expected 'Train accuracy:' in output, got: {out!r}"
    assert "Test accuracy:"  in out, f"Expected 'Test accuracy:' in output, got: {out!r}"

    # check that the classifier is a RandomForestClassifier
    # and that it has 2 classes
    from sklearn.ensemble._forest import RandomForestClassifier
    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_classes_ == 2
