% AU-Typen, welche möglich sind
au(au4).
au(au6).
au(au9).
au(au17).
au(au25).
au(au26).

% CNN-/PFR-Fakten: nur die Trainings fakten, keine Test daten inkludiert
% video/1, frame/2, original_frame/3, time_s/3, active/3

% F1 liegt vor F2 im selben Video -> für zeitliche Analysen
before(V,F1,F2) :-
    frame(V,F1),
    frame(V,F2),
    F1 < F2.

% F2 kommt direkt nach F1 im selben Video
next(V,F1,F2) :-
    frame(V,F1),
    frame(V,F2),
    F2 = F1 + 1.

% Zwei AUs sind im selben Frame aktiv
coactive(V,AU1,AU2,F) :-
    active(V,AU1,F),
    active(V,AU2,F),
    AU1 != AU2.

% AU1 ist irgendwann vor AU2 aktiv
au_before(V,AU1,AU2) :-
    active(V,AU1,F1),
    active(V,AU2,F2),
    before(V,F1,F2).

% AU2 folgt innerhalb von 3 Frames auf AU1
au_followed_within_3(V,AU1,AU2) :-
    active(V,AU1,F1),
    active(V,AU2,F2),
    before(V,F1,F2),
    F2 <= F1 + 3.

% AU2 folgt innerhalb von 5 Frames auf AU1
au_followed_within_5(V,AU1,AU2) :-
    active(V,AU1,F1),
    active(V,AU2,F2),
    before(V,F1,F2),
    F2 <= F1 + 5.

% Zwei AUs überlappen irgendwann im Video
overlap(V, AU1, AU2) :-
    active(V, AU1, F),
    active(V, AU2, F),
    AU1 != AU2.

% Erster aktiver Frame einer AU im Video
first_active(V, AU, F) :-
    active(V, AU, F),
    not earlier_active(V, AU, F).

earlier_active(V, AU, F) :-
    active(V, AU, F),
    active(V, AU, F2),
    F2 < F.

% AU1 startet vor AU2
starts_before(V, AU1, AU2) :-
    first_active(V, AU1, F1),
    first_active(V, AU2, F2),
    F1 < F2,
    AU1 != AU2.

% Anzahl aktiver Frames pro AU
exists_between(V, AU, F1, F2) :-
    active(V, AU, F1),
    active(V, AU, F3),
    active(V, AU, F2),
    F1 < F3,
    F3 < F2.

next_active(V, AU, F1, F2) :-
    active(V, AU, F1),
    active(V, AU, F2),
    F1 < F2,
    not exists_between(V, AU, F1, F2).

active_rank(V, AU, F, 1) :-
    first_active(V, AU, F).

active_rank(V, AU, F2, N+1) :-
    active_rank(V, AU, F1, N),
    next_active(V, AU, F1, F2),
    N < 10.

duration_long(V, AU) :-
    active_rank(V, AU, _, 10).
