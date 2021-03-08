def unary!(v)
    if v then
        0
    else
        1;

def unary-(v)
    0-v;

def binary> 10 (LHS RHS)
    RHS < LHS

def binary| 5 (LHS RHS)
    if LHS then
        1
    else if RHS then
        1
    else
        0;

def binary& 6 (LHS RHS)
    if !LHS then
        0
    else
        !!RHS;

def binary = 9 (LHS RHS)
    !(LHS < RHS | RHS < LHS);

def binary : 1 (x y) y;

extern putchard(char);

def printdensity(d)
    if d > 8 then
        putchard(32)
    else if d > 4 then
        putchard(46)
    else if d > 2 then
        putchard(43)
    else
        putchard(42);

printdensity(1) : printdensity(2) : printdensity(3) : printdensity(4) : printdensity(9) :
printdensity(10)
