from lisp import *

def main():
    import os
    plus = R('+')
    inc = E('fn', ['x'],
      (plus, 'x', 1))
    lmap = R('map')
    print E(lmap, lambda x: x+1, [3, 4, 5])


if __name__ == '__main__':
    main()

