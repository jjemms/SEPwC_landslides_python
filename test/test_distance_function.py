import pytest
import sys
import rasterio
import numpy as np
sys.path.insert(0,"../")
sys.path.insert(0,"./")
from proximity import *
from pylint.lint import Run
from pylint.reporters import CollectingReporter
from dataclasses import asdict

class TestProximity():
    
    def test_proximity(self):

        raster = rasterio.open("test/data/rasterised.tif")
        distance = proximity(raster, raster.read(1), 1)
        assert(77.10382 < np.max(distance) < 77.10383)
        assert(-1e-8 < np.min(distance) < 1e-8)        

    def test_lint(self):
        files =  ["proximity.py"]
        #pylint_options = ["--disable=line-too-long,import-error,fixme"]
        pylint_options = []

        report = CollectingReporter()
        result = Run(
                    files,
                    reporter=report,
                    exit=False,
                )
        score = result.linter.stats.global_note
        nErrors = len(report.messages)

        print("Score: " + str(score))
        line_format = "{path}:{line}:{column}: {msg_id}: {msg} ({symbol})"
        for error in report.messages:
            print(line_format.format(**asdict(error)))   

        assert nErrors == 0
        assert score > 9

