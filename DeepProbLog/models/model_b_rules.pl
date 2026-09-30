% Modell B: FastLAS-gelernte Regelstruktur (aus ILP/learned_rules/)
% Verwendet nur die AUs, die in den gelernten Regeln vorkommen: au4, au6, au9, au17

t(0.5) :: pain_weight1.

t(0.5) :: disgust_weight1.
t(0.5) :: disgust_weight2.
t(0.5) :: disgust_weight3.

% Pain (aus learned_rules/pain_oracle_tiny.lp) ---
pain(V) :- duration_long(V,au26), video(V), pain_weight1.

% Disgust (aus learned_rules/disgust_oracle_train_rule1/2/3.lp) ---
disgust(V) :- overlap(V,au9,au6), video(V), disgust_weight1.
disgust(V) :- overlap(V,au6,au4), video(V), disgust_weight2.
disgust(V) :- duration_long(V,au4), duration_long(V,au17), video(V), disgust_weight3.