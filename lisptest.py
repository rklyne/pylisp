import unittest
import lisp

class LispTest(unittest.TestCase):
    def setUp(self):
        "Prepare a StringReader to read in s-expressions during tests."
        self.reader = lisp.StringReader()

    def tearDown(self):
        "Check that there is no unconsumed input, then reset the environment."
        self.assertEqual(self.reader.get_remaining_input(), "")
        lisp.reset()

    def get_expr(self, text):
        "Use the reader to convert text to an s-expression"
        self.reader.provide_input(text)
        expr = self.reader.get_expr()
        self.assertEqual(self.reader.get_remaining_input(), "")
        return expr

    def get_eval(self, text):
        "Pass the textual expression through the reader, then evaluate it."
        reader_expr = self.get_expr(text)
        return lisp.lisp_eval(reader_expr)

    def assertExpr(self, text, expr_tuple):
        "Assert that some text is parsed as a particular s-expression."
        reader_expr = self.get_expr(text)
        self.assertEqual(reader_expr, expr_tuple)

    def assertEval(self, text, expr_tuple):
        "Assert that some text is evaluated to a particular s-expression."
        result_expr = self.get_eval(text)
        self.assertEqual(result_expr, expr_tuple)

    def assertQuoteEval(self, text, expr_tuple):
        result_expr = self.get_eval(text)
        if type(result_expr) is lisp.QuoteWrapper:
            result_expr = result_expr.content
        self.assertEqual(result_expr, expr_tuple)

class ReaderTest(LispTest):
    # 1. A Symbol
    def test_symbol(self):
        expr = self.get_expr("abc")
        self.assert_(isinstance(expr, lisp.Symbol))
        self.assertEqual(expr, "abc")

    # 2. A number, int or float.
    def test_int(self):
        self.assertExpr("1235", 1235)
    def test_float(self):
        self.assertExpr("1235.5", 1235.5)

    # 3. A string
    def test_string(self):
        expr = self.get_expr('"abc"')
        self.assert_(isinstance(expr, str))
        self.assertEqual(expr, "abc")

    def test_string_escaping(self):
        expr = self.get_expr('"abc\\"abc"')
        self.assert_(isinstance(expr, str))
        self.assertEqual(expr, 'abc"abc')

    # 4. A comment
    def test_comment(self):
        self.assertExpr("; comment", None)

    # 5. A sequence
    def test_sequence(self):
        expr = self.get_expr("[abc 123]")
        self.assert_(isinstance(expr, lisp.Sequence))
        self.assertEqual(expr[0], "abc")
        self.assertEqual(expr[1], 123)

    # 6. A parenthesised whitespace separated list of s-expressions
    def test_s_expression(self):
        expr = self.get_expr("(abc 123)")
        self.assert_(isinstance(expr, lisp.SExpr))
        self.assertEqual(expr[0], "abc")
        self.assertEqual(expr[1], 123)

    def test_bad_expr1(self):
        self.bad_expr_test("|")
    def test_bad_expr2(self):
        self.bad_expr_test("\\")
    def bad_expr_test(self, expr):
        try:
            self.get_expr(expr)
        except lisp.ReaderError:
            pass
        else:
            self.fail("Expected a lisp.ReaderError")

    def test_quoting(self):
        self.assertQuoteEval("`(a b c)", ('a', 'b', 'c'))

class EvalTest(LispTest):
    def test_addition(self):
        self.assertEval("(+ 3 4)", 7)

    def test_quote(self):
        self.assertEval("(quote a)", "a")
        self.assertEval("'a", "a")
        self.assertEval("'(2 3 4)", (2, 3, 4))
        self.assertEval("'(2 3 (4))", (2, 3, (4, )))
        self.assertQuoteEval("`(1 ~2)", (1, 2))
        self.assertQuoteEval("`(1 ~(+ 3 2))", (1, 5))
        self.assertQuoteEval("`(1 ~(+ 3 2) ~(+ 3 2) ~(+ 3 2))", (1, 5, 5, 5))
        self.assertQuoteEval("`(1 (2 ~(+ 3 2)))", (1, (2, 5)))

    def test_truth(self):
        self.assertEval("t", True)

    def test_def(self):
        self.assertEval("(def a 3)", 3)

    def test_defn(self):
        """Test defining and calling functions"""
        expr = self.get_eval("(def f (fn [x] (+ x 1)))")
        self.assertEval("(f 2)", 3)

    def test_no_function_args(self):
        "Test defining and calling a funciton with no parameters"
        self.get_eval("(defn g [] 2)")
        self.assertEval("(g)", 2)

    def test_function_parameters(self):
        "Test variable length function arguments"
        expr = self.get_eval("(def f (fn [x &rest] rest))")
        self.assertEval("(f 2 3 4 5)", (3, 4, 5))

    def test_conditional_branching(self):
        self.assertEval("(if t 2 3)", 2)
        self.assertEval("(if nil 2 3)", 3)
        self.assertEval("(if nil 2)", None)

    def test_if_non_eval(self):
        self.get_eval("(def x 2)")
        self.assertEval("(if nil y x)", 2)

    def test_string(self):
        self.assertEval('"hi"', "hi")
        self.assertEval('(if t "yes" "no")', "yes")

    def test_defmacro(self):
        """Test defining macros"""

        # A macro that runs code when some variable equals 7.
        code = """
        (defmacro when7 [test &body]
          (list 'if (list 'eq test '7) (cons 'progn body)))
        """
        self.get_eval(code)
        self.assertEval("(when7 7 2)", 2)
        self.assertEval("(when7 3 2)", None)
        self.assertEval("(when7 (+ 3 4) 2)", 2)
        self.assertEval("(when7 '(+ 3 4) 2)", None)

    def test_let(self):
        self.get_eval("(def a 1)")
        self.get_eval("(defn g [x] (let [a 2] (+ x (h a))))")
        self.get_eval("(let [a 3] (defn h [x] (+ x a)))")
        self.assertEval("(g 10)", 15)

    def test_recursion(self):
        "Factorial calculation is the traditional recursion test..."
        self.get_eval("(defn factorial [n] (if (= n 1) 1 (* n (factorial (- n 1)))))")
        self.assertEval("(factorial 4)", 24)

    def test_special_cases(self):
        self.assertEval("()", ())
        self.assertEval("nil", None)
        self.assertEval("t", True)

    def test_eval_str(self):
        self.assertEval("(eval-str \"(+ 2 3)\")", 5)
        self.get_eval("(def x 1)")
        self.assertEval("(eval-str \"(+ x 3)\")", 4)

    def test_head_tail(self):
        self.assertEval("(head '(2 3))", 2)
        self.assertEval("(head '(() 3))", ())
        self.assertEval("(tail '(2 3))", (3, ))
        self.assertEval("(tail '(() 3))", (3, ))
        self.assertEval("(tail '(2 (3)))", ((3, ), ))

    def test_concat(self):
        self.assertEval("(concat '(2 3) '(4 5))", (2, 3, 4, 5))

    def test_flatten(self):
        self.assertEval("(flatten '(4 5))", (4, 5))
        self.assertEval("(flatten '((4 5)))", (4, 5))
        self.assertEval("(flatten '(2 3 (4 5)))", (2, 3, 4, 5))
        self.assertEval("(flatten '(2 3 (4 5) 6))", (2, 3, 4, 5, 6))
        self.assertEval("(flatten '((2 3) (4 5)))", (2, 3, 4, 5))

    def test_dot(self):
        l = lisp.Lisp()
        class C(object): pass
        o = C()
        o.x = 1
        l.E('.', l.Q(o), l.string("x"), 2)
        self.assert_(o.x == 2)

    def test_bool(self):
        self.assertEval("(and 1 1 1 1)", True)
        self.assertEval("(and 1 1 1 1 0)", None)
        self.assertEval("(and 1 1 ())", None)
        self.assertEval("(and 1 1 '(a))", True)
        self.assertEval("(and 0 x)", None)
        self.assertEval("(or 0 0 0 0)", None)
        self.assertEval("(or 0 0 0 0 1)", True)
        self.assertEval("(or 1 x)", True)

class EnvTest(LispTest):
    basic_env_keys = [
        '+',
        'seq',
        'cons',
    ]

    core_pyl_keys = [
        'defn',
        'when',
    ]

    def test_lookup(self):
        env = lisp.LispEnv({})
        sym_a = lisp.Symbol('a')
        env[sym_a] = 1
        self.assertEqual(env[sym_a], 1)

    def test_parent_lookup(self):
        env = lisp.LispEnv({})
        sym_a = lisp.Symbol('a')
        sym_b = lisp.Symbol('b')
        env[sym_a] = 1
        env2 = lisp.LispEnv({sym_a: 2}, parent=env)
        self.assertEqual(env2[sym_a], 2)

        env3 = lisp.LispEnv({}, parent=env2)
        env4 = lisp.LispEnv({sym_a: 4}, parent=env3)
        self.assertEqual(env4[sym_a], 4)

        env5 = lisp.LispEnv({sym_b: 4}, parent=env3)
        env6 = lisp.LispEnv({sym_b: 4}, parent=env5)
        self.assert_(env6.has_key(sym_b), "Expected to find 'b' in env")
        self.assert_(env6.has_key(sym_a), "Expected to find 'a' in env")
        self.assertEqual(env6[sym_b], 4)
        self.assertEqual(env6[sym_a], 2)

    def test_basic_env(self):
        env = lisp.get_basic_env()
        for key in self.basic_env_keys:
            self.assert_(env.has_key(key), msg="Expected basic env to have key '%s'" % key)
            self.assertNotEqual(env.lookup(key), None)

    def test_builtin_env(self):
        env = lisp.get_basic_env()
        for key in __builtins__.__dict__:
            self.assert_(env.has_key(key))

    def test_simple_env(self):
        env = lisp.get_simple_env()
        keys = self.basic_env_keys + self.core_pyl_keys
        for key in keys:
            self.assert_(env.has_key(key), msg="Expected simple env to have key '%s'" % key)
            self.assertNotEqual(env.lookup(key), None)

    def test_stackable(self):
        env = lisp.LispEnv({})
        sym_a = lisp.Symbol('a')
        sym_b = lisp.Symbol('b')
        env[sym_a] = 1
        self.assertEqual(env[sym_a], 1)

        env = lisp.make_stackable_env(env)
        env.define(sym_b, 2)
        def assert_ab(a, b):
            self.assertEqual(env.lookup(sym_a), a)
            self.assertEqual(env.lookup(sym_b), b)
        assert_ab(1, 2)

        new_env = {
            sym_a: 3,
        }
        new_env = env.new_scope(new_env)
        assert_ab(1, 2)
        try:
            env.push(new_env)
            assert_ab(3, 2)
            env.define(sym_b, 4)
            assert_ab(3, 4)
        finally:
            env.pop()

        assert_ab(1, 2)

class ScopeTest(LispTest):
    def test_fn_scope(self):
        self.get_eval("(def a 1)")
        self.get_eval("(defn g [a] a)")
        self.assertEval("(g 3)", 3)

    def test_fn_scope_privacy(self):
        self.get_eval("(def a 1)")
        self.get_eval("(defn g [x] (+ x (h a)))")
        self.get_eval("(defn h [a] (+ x a))")
        try:
            self.assertEval("(g 30)", 41)
        except lisp.LookupError, le:
            self.assertEquals(le.args[0], 'x')
        else:
            self.fail("Expected LookupError('x')")
        self.get_eval("(def x 10)")
        self.assertEval("(g 30)", 41)

    def test_let(self):
        self.get_eval("(def a 1)")
        self.get_eval("(defn g [x] (let [a 2] (+ x (h a))))")
        self.get_eval("(defn h [x] (+ x a))")
        self.assertEval("(g 10)", 13)

    def test_binding_as_let(self):
        self.get_eval("(def a 1)")
        self.assertEval("(binding [a 2] a)", 2)

    def test_binding(self):
        self.get_eval("(def a 1)")
        self.get_eval("(defn g [] a)")
        self.assertEval("(binding [a 2] (g))", 2)

    def test_deeper_binding(self):
        self.get_eval("(def a 1)")
        self.get_eval("(defn g [x] (binding [a 2] (+ x (h a))))")
        self.get_eval("(defn h [x] (+ x a))")
        self.assertEval("(g 10)", 14)

L = lisp.Lisp()
class IntegrationTest(unittest.TestCase):
    def test_Q(self):
        q = L.Q(3)
        assert type(q) is lisp.SExpr
        assert q[1] == 3
        assert q[0] == lisp.QUOTE_SYMBOL

    def test_S(self):
        assert L.SExpr(2) == (2, )
        assert L.SExpr(2, 3) == (2, 3, )
        assert L.SExpr(2, ('+', 3, 5)) == (2, (L.sym('+'), 3, 5), )

        plus = L.R('+')
        assert L.SExpr(2, (plus, 3, 5)) == (2, (L.R('+'), 3, 5), )

    def test_E(self):
        plus = lambda x, y: x + y
        assert L.E(plus, 3, 5) == 8

    def test_R(self):
        plus = L.R('+')
        assert callable(plus), plus
        result = plus(2, 3)
        assert result == 5, result

    def test_def(self):
        L('def', 'C', 5)
        assert L.R('C') == 5

    def test_fn(self):
        f = L.E('fn', ['a'], (L.R('+'), 'a', 1))
        assert callable(f), f
        assert f(3) == 4, f

    def test_defn(self):
        f = L.E('defn', 'f', [], 2)
        g = L.E('defn', 'g', [], ('+', (f,), 3))
        assert L.E('g') == 5
        assert g() == 5
        h = L.E('defn', 'h', ['a'], ('+', ('f',), 'a'))
        assert h(6) == 8

    def test_recursion(self):
        def decrement(num):
            return num - 1
        g = L.E('defn', 'fact', ['a'],
            ('if', ('=', 'a', 0),
                1,
                ('*', 'a', ('fact', (decrement, 'a'))),
            )
        )

        assert g(0) == 1
        assert g(5) == (5*4*3*2)

    def test_strings(self):
        yes_val = 'yes'
        no_val = 'no'
        fn = L.E('fn', ['a'], ('if', 'a', L.string(yes_val), L.string(no_val) ))
        assert fn(True) == yes_val
        assert fn(False) == no_val

if __name__ == '__main__':
    unittest.main()

