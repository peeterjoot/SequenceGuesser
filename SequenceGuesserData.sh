#! /usr/local/bin/wolframscript

t = Table[{RandomReal[{0.8, 1.05}], RandomReal[{0.8, 1.05}],
  RandomReal[{0, 2}], RandomReal[{0, 2}]}, {i, 0, 10}];

Export["trainingdata.txt", t]  (* ugly: missing newline *)

(* could do:
Export["trainingdata.txt", StringJoin[ExportString[t, "CSV"], "\n"]]

-- but that's doesn't have the {} delimited lines of the original above.
*)
