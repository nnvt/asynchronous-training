"""Digitize the median curves of the paper's Fig. 1 from a 600-dpi render.

Usage: python -I tools/digitize_fig1.py <panel.png> <panel-name> > out.json
Pixel calibration (frame, tick rows/cols) was measured on renders made with
  pdftoppm -png -r 600 -f 16 -l 16 -x 900  -y 660 -W 1700 -H 1120 iclr.pdf hr_a   (MNIST)
  pdftoppm -png -r 600 -f 16 -l 16 -x 2580 -y 660 -W 1640 -H 1120 iclr.pdf hr_b   (FMNIST)
"""
import json
import sys

import numpy as np
from PIL import Image

CAL = {  # x0 pixel, px per time unit, (row of 10^e0, e0), px per decade, frame, legend box to skip
    "mnist": dict(x0=142.5, xs=(1525.0 - 142.5) / 200000, y_ref=(180.5, 3), ppd=(866.5 - 180.5) / 6,
                  frame=(142, 1527, 37, 1027), legend=(1000, 1527, 37, 260)),
    "fmnist": dict(x0=178.5, xs=(1508.5 - 178.5) / 350000, y_ref=(64.5, 5), ppd=(901.5 - 64.5) / 8,
                   frame=(178, 1604, 37, 1027), legend=(1080, 1604, 37, 260)),
}
COLORS = {"ringleader": (192, 0, 0), "malenia": (0, 100, 100), "ia2sgd": (186, 132, 12)}


def main(path, panel):
    c = CAL[panel]
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    L, R, Tp, B = c["frame"]
    out = {}
    for m, col in COLORS.items():
        d = np.sqrt(((im - np.array(col)) ** 2).sum(2))
        mask = d < 40
        lx0, lx1, ly0, ly1 = c["legend"]
        mask[ly0:ly1, lx0:lx1] = False
        mask[:, :L + 3] = False
        mask[:, R - 2:] = False
        mask[:Tp + 3] = False
        mask[B - 2:] = False
        ts, vs = [], []
        for x in range(L + 3, R - 2):
            rows = np.nonzero(mask[:, x])[0]
            if len(rows) == 0 or rows.max() - rows.min() > 40:  # skip marker / ambiguous columns
                continue
            y = rows.mean()
            t = (x - c["x0"]) / c["xs"]
            v = 10 ** (c["y_ref"][1] - (y - c["y_ref"][0]) / c["ppd"])
            ts.append(t)
            vs.append(v)
        out[m] = dict(t=ts, v=vs)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
