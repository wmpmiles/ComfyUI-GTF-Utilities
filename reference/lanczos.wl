(* ::Package:: *)

(* ::Input:: *)
(*L[a_, x_]:=Piecewise[{{Sinc[Pi*x]Sinc[Pi*x/a], -a<x<a}},0]*)


(* ::Input:: *)
(*Plot3D[L[3,x]*L[3,y],{x,-4,4},{y,-4,4},WorkingPrecision->20,PlotRange->Full,MaxRecursion->10]*)


(* ::Input:: *)
(*Plot3D[L[3,Sqrt[x^2+(y)^2]],{x,-4,4},{y,-4,4},PlotRange->Full,WorkingPrecision->10,MaxRecursion->10,PlotPoints->25]*)


(* ::Input:: *)
(*NIntegrate[L[20,x],{x,-Infinity,Infinity}]*)


(* ::Input:: *)
(*Plot[L[3,x],{x,-4,4},PlotRange->Full]*)
