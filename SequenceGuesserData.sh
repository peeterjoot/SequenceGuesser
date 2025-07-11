#! /usr/local/bin/wolframscript

t = Table[{RandomReal[{0.8, 1.05}], RandomReal[{0.8, 1.05}],
  RandomReal[{0, 2}], RandomReal[{0, 2}]}, {i, 0, 10}];

Export["trainingdata.json", t]
