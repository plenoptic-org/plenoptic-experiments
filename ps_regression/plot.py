#!/usr/bin/env python3

import sys

sys.path.append("/mnt/home/wbroderick/plenoptic/tests")
from utils import visualize_ps_regression

visualize_ps_regression(sys.argv[1], sys.argv[2])
