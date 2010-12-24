import unittest
import lisp

class LispTest(unittest.TestCase):
    def setUp(self):
        self.reader = lisp.StringReader()

    def tearDown(self):
        self.assertEqual(self.reader.get_remaining_input(), "")

    def get_expr(self, text):
        self.reader.provide_input(text)
        expr = self.reader.get_expr()
        self.assertEqual(self.reader.get_remaining_input(), "")
        return expr

    def assertExpr(self, text, expr_tuple):
        reader_expr = self.get_expr(text)
        self.assertEqual(reader_expr, expr_tuple)

    def assertEval(self, text, expr_tuple):
        reader_expr = self.get_expr(text)
        result_expr = lisp.lisp_eval(reader_expr)
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

if __name__ == '__main__':
    unittest.main()





