% Letzter gezielter Versuch: nur die zwei Merkmale mit groesster
% gemessener Klassentrennung, je einzeln, ohne UND-Verknuepfung.
t(0.5) :: pain_weight1.
t(0.5) :: disgust_weight1.

pain(V) :- duration_long(V,au25), pain_weight1.
disgust(V) :- duration_long(V,au9), disgust_weight1.
