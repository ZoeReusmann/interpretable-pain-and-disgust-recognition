% Modell A: Literaturbasierte Regelstruktur (v3)
% Pain: Kunz et al. (systematic review, 2019), Prkachin (1992), Kunz et al. (2013), Disgust: Ekman & Friesen (1978)
% Hinweis: ursprüngliches Prädikat active(V,au) (P[>=1 Frame aktiv]) wurde
% getestet und aufgrund von Sättigung (Median = 1.0 für alle AUs, ~175
% Frames/Video) verworfen; alle Regeln nutzen daher durchgaengig duration_long.

t(0.5) :: pain_weight1.
t(0.5) :: pain_weight2.

t(0.5) :: disgust_weight1.
t(0.5) :: disgust_weight2.
t(0.5) :: disgust_weight3.
t(0.5) :: disgust_weight4.

% Pain
% Kunz et al. (systematic review) + Prkachin (1992): brow lowering, orbital tightening, nose wrinkling
pain(V) :- duration_long(V,au4), duration_long(V,au6), duration_long(V,au9), pain_weight1.
% Kunz et al. (2013): zusätzlich mouth opening
pain(V) :- duration_long(V,au4), duration_long(V,au6), duration_long(V,au9), duration_long(V,au25), pain_weight2.

% Disgust
% Ekman & Friesen (1978): AU9 als Kernmerkmal
disgust(V) :- duration_long(V,au9), disgust_weight1.
disgust(V) :- duration_long(V,au9), duration_long(V,au17), disgust_weight2.
disgust(V) :- duration_long(V,au9), duration_long(V,au25), disgust_weight3.
% Ekman & Friesen (1978): gemeinsames Auftreten von au9 und au17
disgust(V) :- overlap(V,au9,au17), disgust_weight4.