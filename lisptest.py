import unittest
import lisp

class LispTest(unittest.TestCase):
    def setUp(self):
        self.reader = lisp.StringReader()

    def tearDown(self):
        self.assertEqual(self.reader.get_remaining_input(), "")
        lisp.reset()

    def get_expr(self, text):
        self.reader.provide_input(text)
        expr = self.reader.get_expr()
        self.assertEqual(self.reader.get_remaining_input(), "")
        return expr

    def get_eval(self, text):
        reader_expr = self.get_expr(text)
        return lisp.lisp_eval(reader_expr)

    def assertExpr(self, text, expr_tuple):
        reader_expr = self.get_expr(text)
        self.assertEqual(reader_expr, expr_tuple)

    def assertEval(self, text, expr_tuple):
        result_expr = self.get_eval(text)
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

class EvalTests(LispTest):
    def test_addition(self):
        self.assertEval("(+ 3 4)", 7)

    def test_quote(self):
        self.assertEval("(quote a)", "a")
        self.assertEval("'a", "a")

    def test_truth(self):
        self.assertEval("t", True)

    def test_def(self):
        self.assertEval("(def a 3)", 3)

    def test_defn(self):
        """Test defining and calling functions"""
        expr = self.get_eval("(def f (fn [x] (+ x 1)))")
        self.assertEval("(f 2)", 3)

    def test_conditional_branching(self):
        self.assertEval("(if t 2 3)", 2)
        self.assertEval("(if nil 2 3)", 3)
        self.assertEval("(if nil 2)", None)

    def test_defmacro(self):
        """Test defining macros"""

        # A macro that runs code when
        code = """
        (defmacro when7 [test &body]
          (seq 'if (seq 'eq test '7) (cons 'progn body)))
        """
        self.get_eval(code)
        self.assertEval("(when7 7 2)", 2)
        self.assertEval("(when7 3 2)", None)

class EnvTest(LispTest):
    basic_env_keys = [
        '+',
        'seq',
        'cons',
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
            self.assertNotEqual(env[key], None)

    def test_simple_env(self):
        env = lisp.get_simple_env()
        for key in self.basic_env_keys:
            self.assert_(env.has_key(key), msg="Expected basic env to have key '%s'" % key)
            self.assertNotEqual(env[key], None)

if __name__ == '__main__':
    unittest.main()

