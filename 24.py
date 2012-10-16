
import lisp

def g24():
    l = lisp.Lisp()

    import random
    def rand(top):
        return random.randint(1, top)

    l.E('defn', 'gen-new-game-nums', ['amount'], ('repeatedly', 'amount', (rand, 9)))

    def seq_eq(s1, s2):
        l1 = list(s1)
        l2 = list(s2)
        l1.sort()
        l2.sort()
        return l1 == l2

    def valid_input_1(user_input):
        """checks whether the expression is somewhat valid prefix notation
        (+ 1 2 3 4) (+ 3 (+ 4 5) 6)
        this is done by making sure the only contents of the list are numbers operators and brackets
        flatten gets rid of the brackets, so we just need to test for operators and integers after that"""
        try:
            expr = l.E('flatten', ('read-string', l.string(user_input)))
        except Exception, e:
            raise
            return False
        for item in expr:
            if isinstance(item, int):
                continue
            try:
                if int(item):
                    continue
            except (TypeError, ValueError):
                pass
            found_sym = False
            for name in [
                '+',
                '-',
                '*',
                '/',
            ]:
                if name == item:
                    found_sym = True
                    break
            if found_sym:
                continue
            raise RuntimeError(item, type(item), expr, user_input)
            return False
        return True


    valid_input = l.E('fn', ['i'], ('if', ('=', 'i', l.string('q')),
        False,
        (valid_input_1, 'i'),
    ))

    l.E('defn', 'game-numbers-and-user-input-same?',
#      "input form: (+ 1 2 (+ 3 4))
#    tests to see if the numbers the user entered are the same as the ones given to them by the game"
      ['game-nums', 'user-input'],
      (seq_eq, 'game-nums', (filter, 'integer?', ('flatten', 'user-input'))))

    l.E('defn', 'win', [], ('println', l.string("you won the game!\n")))
    l.E('defn', 'lose', ['input', 'game-numbers', 'goal'],
      ('println', l.string("you guessed wrong, or your input was not in prefix notation. eg: '(+ 1 2 3 4)'\n")),
      ('if', ('not', (valid_input, 'input')), ('println', l.string("Your input is invalid")),
        ('if', ('not', ('game-numbers-and-user-input-same?', 'game-numbers', ('read-string', 'input'))),
          ('println', l.string("you reused numbers")),
          'nil')))
    l.E('defn', 'game-start', ['goal', 'game-numbers'], ('progn',
                                           ('println', l.string("Your numbers are "), 'game-numbers'),
                                           ('println', l.string("Your goal is "), 'goal'),
                                           ('println', l.string("Use the numbers and +*-/ to reach your goal\n")),
                                           ('println', l.string("'q' to Quit\n"))))

    l.E('defn', 'play-game',
#      "typing in 'q' quits.
#       to play use (play-game) (play-game 24) or (play-game 24 '(1 2 3 4)"
      [], ('play-game-1', 24))
    l.E('defn', 'play-game-1', ['goal'], ('play-game-2', 'goal', ('gen-new-game-nums', 4))),
    l.E('defn', 'play-game-2', ['goal', 'game-numbers'],
       ('game-start', 'goal', 'game-numbers'),
       ('let', ['input', ('read-line',)],
         ('if', ('and', (valid_input, 'input'),
                  ('game-numbers-and-user-input-same?', 'game-numbers', ('read-string', 'input')),
                  ('=', 'goal', ('eval-str', 'input'))),
           ('win',),
           ('when', ('not', ('=', 'input', l.string("q"))),
             ('progn', ('lose', 'input', 'game-numbers', 'goal'), ('println', l.string("you got "), ('eval-str', 'input')), ('play-game-2', 'goal', 'game-numbers'))))))

    print l.E('game-numbers-and-user-input-same?', l.Q((2,3,4,5)), ('read-string', l.string("(+ 3 2 4 5)")))
    print l.E('=', 24, ('eval-str', l.string("(+ 12 12)")))

    l.E('play-game')

if __name__ == '__main__':
    g24()
