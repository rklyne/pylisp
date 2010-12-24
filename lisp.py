#
# PyLisp / Pyle (Python List Expressions)
#
# Ronan Klyne, http://github.com/users/rklyne
#

# imports
import sys
import re


# Exceptions
class LispError(Exception): pass
class LispRuntimeError(RuntimeError): pass
class ReaderError(LispError): pass
class ExpressionEvalError(LispError): pass
class LookupError(LispError): pass


# s-expression type
class SExpr(tuple): pass
class Sequence(tuple): pass

class Symbol(str): pass

# enivronment
class LispEnv(dict): pass

class LispFunc(object):
    """In general, a Python function is a PyLisp function, and a PyLisp function
    needs this wrapper to make it
    """
    def __init__(self, expr, bindings, env):
        self.expr = expr
        self.bindings = bindings
        self.def_env = env
    def __call__(self, *args):
        # TODO: Implement this so that Lisp code is Python callable.
        # (Python code is Lisp callable, so this is also required to make Lisp
        # functions Lisp callable ;))
        raise NotImplementedError

env_stack = []
def lisp_apply(f, args, env):
    assert callable(f), f
    try:
        # XXX: Not thread safe except in pure PyLisp code
        env_stack.append(env)
        return f(*args)
    finally:
        assert env_stack.pop() is env

def build_basic_env():
    _no_value = object()
# Maths functions
    def plus(arg0, *args):
        return arg0 + sum(args)
    def times(arg0, *args):
        result = arg0
        for arg in args:
            result *= arg
        return result
    def minus(l, r):
        return l - r
    def divide(l, r):
        return l / r

# String functions
    def string(arg0, *args):
        res = str(arg0)
        for arg in args:
            res += str(arg)
        return res
# IO functions
    def print_func(*args):
        for arg in args:
            print arg,
        print

# System functions
    def assert_func(self, truth, message=_no_value):
        if message is _no_value:
            assert truth
        else:
            assert truth, message

# The environment, with Lisp names.
    env = LispEnv(**{
        't': True,
        'f': False,
        '+': plus,
        '-': minus,
        '*': times,
        '/': divide,
        'print': print_func,
        'str': string,
        'assert': assert_func,
        'apply': apply,
    })

    return env

basic_env = build_basic_env()

def resolve_def(var, env):
    if not env.has_key(var):
        raise LookupError(var)
    return env[var]


# The Reader

class Reader(object):
    _prompt = 'Lisp --> '
    _nl_expr = re.compile(r"[\n\r]")
    _ws_expr = re.compile(r"[\t\n\r, ]")
    _ws_char = ' '
    _debug = True

    def __init__(self, input_func=raw_input):
        self._raw_input = input_func
        self._buffer = ''
        self._last_form = None

    def _input(self):
        data = ''
        while not data:
            data = self._raw_input(self._prompt)
        return data + "\n" # This nl gets stripped by python :-(
    def _read(self):
        try:
            data = str(self._input())
        except EOFError:
            return ""
        # Replace all whitespace chars with the one true ws-char.
#        data = self._ws_expr.sub(self._ws_char, data)
        self._buffer += data

    def _ensure_data(self, error=False):
        if not self._buffer:
            if self._read() == "":
                if error:
                    raise EOFError

    def _read_until(self, str_or_expr, matches=True):
        if isinstance(str_or_expr, str):
            expr = re.compile(str_or_expr)
        else:
            expr = str_or_expr
        # TODO: This could be much more efficient.
        data = ''
        while self.peek_char() and (expr.match(self.peek_char()) is not None) != matches:
            data += self.get_char()
        return data
    def _buffer_drop(self, count=1):
        if self._debug and self._last_form is not None:
            self._last_form += self._buffer[:count]
        self._buffer = self._buffer[count:]
    def _drop_until(self, expr, matches=True):
        # TODO: This could be much more efficient.
        while self.peek_char() and (expr.match(self.peek_char()) is not None) == matches:
            self._buffer_drop()
    def _drop_ws(self):
        self._drop_until(self._ws_expr)
    def _drop_until_newline(self):
        self._drop_until(self._nl_expr, False)

    def get_char(self):
        self._ensure_data(error=True)
        c = self._buffer[0]
        self._buffer_drop()
        return c

    def peek_char(self):
        self._ensure_data()
        if not self._buffer:
            return ""
        return self._buffer[0]

    def get_non_ws_char(self):
        self._drop_ws()
        return self.get_char()

    def get_expr(self,
        digits='0123456789',
        symbol_start_chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_+-/*!?$^~#',
        symbol_chars=''
    ):
        """Reads a form and returns one full s-expression. An s-expression may be:
        0. A reader macro: These start with "#"
        1. A Symbol
        2. A number, int or float.
        3. A string
        4. A comment
        5. A sequence
        6. A parenthesised whitespace separated list of s-expressions
        """
        self._last_form = ''
        c = self.get_non_ws_char() # First char.

        # 0. Reader macros
        if c == '#':
            raise ReaderError, "No such macro"
        # 1. A Symbol
        if c in symbol_start_chars:
            sym = c
            c = self.peek_char()
            while c and (c in symbol_chars or c in symbol_start_chars):
                sym += c
                self.get_char()
                c = self.peek_char()
            return Symbol(sym)
        # 2. A number
        if c in digits:
            n = c
            # TODO: Add float support
            while self.peek_char() and self.peek_char() in digits:
                n += self.get_char()
            if self.peek_char() == '.':
                n += self.get_char()
                while self.peek_char() and self.peek_char() in digits:
                    n += self.get_char()
                return float(n)
            return int(n)
        # 3. A string
        if c == '"':
            s = self._read_until('"')
            if s:
                while s[-1] == "\\":
                    s += self.get_char()
                    s += self._read_until('"')
            # Consume the ending quote
            self.get_char()
            # Unescape
            s = s.replace('\\n', '\n')
            s = s.replace('\\t', '\t')
            s = s.replace('\\r', '\r')
            s = s.replace('\\"', '"')
            s = s.replace('\\\\', '\\')
            return s
        # 4. Comment
        if c == ';':
            self._drop_until_newline()
            return None
        nested_expr_params = None
        # 5. A sequence
        if c == '[':
            nested_expr_params = ']', Sequence
        # 6. Parenthesised list of s-expressions
        if c == '(':
            nested_expr_params = ')', SExpr
        # The real work for 5. and 6.
        if nested_expr_params:
            closing_char, expr_class = nested_expr_params
            self._drop_ws()
            s_expr = []
            while self.peek_char() and self.peek_char() != closing_char:
                s_expr.append(self.get_expr())
                self._drop_ws()
            # consume the closing char
            self.get_char()
            return expr_class(s_expr)
        # Something unexpected
        raise ReaderError("Unexpected form", c, self._last_form, self._buffer)

class FileReader(Reader):
    _CHUNK = 256
    def __init__(self, filename):
        self._fp = open(filename, 'rb')
        super(FileReader, self).__init__()
    def _input(self):
        data = self._fp.read(self._CHUNK)
        if data: return data
        raise EOFError
    def __iter__(self):
        try:
            while True:
                expr = self.get_expr()
                if expr:
                    yield expr
        except EOFError:
            pass

class StringReader(Reader):
    _CHUNK = 256
    def __init__(self, text=""):
        self._text = text
        super(StringReader, self).__init__()
    def _input(self):
        ret = self._text
        if ret == "":
            raise EOFError
        self._text = ""
        return ret

    def clear_input(self):
        self._text = ""

    def provide_input(self, text):
        self._text += text

    def get_remaining_input(self):
        return self._text


# The Evaluator (the main bit)
def lisp_eval(expr, env=basic_env):
    """PyLisp Evaluator.
    Reserved words:
    * def
    * progn
    * nil
    """
    type_e = type(expr)
    if type_e is Symbol:
        # 'nil' can't be overridden.
        # For contrast 't' and 'f' (True and False) can be.
        if   expr == 'nil': return None
        else: return resolve_def(expr, env)
    elif type_e is int: return expr
    elif type_e is float: return expr
    elif isinstance(expr, tuple):
        if len(expr) == 0:
            return None
        f, r = expr[0], expr[1:]
        type_f = type(f)
        if type_f is Symbol:
            if f == 'def':
                assert len(r) == 2, r
                name = r[0]
                assert type(name) is Symbol, name
                expr = r[1]
                env[name] = lisp_eval(expr)
                return None
            elif f == 'progn':
                ret = None
                while r:
                    ret = lisp_eval(r[0])
                    r = r[1:]
                return ret
            elif f == 'quote':
                raise NotImplementedError("TODO: Implement quote")
            elif f == 'fn':
                assert len(r) >= 2, r
                # A new LispFunc, with bindings and an expression body
                return LispFunc(r[0], r[1:])

            func = resolve_def(f, env)
            return lisp_apply(func, map(lisp_eval, r), env)
        else:
            raise ExpressionEvalError("expression must start with a symbol")
    else:
        raise LispRuntimeError("Cannot evaluate this expression.", expr)

run = lisp_eval


def repl(debug=False):
    """REPL:
    * R ead s-expressions and
    * E valuate them, then
    * P rint the results.
    * L oop."""
    reader = Reader()
    try:
        while 1:
            try:
                expr = reader.get_expr()
                if debug: print "READ:", expr
            except LispError:
                print "READER ERROR:"
                import traceback
                traceback.print_exc()
                continue
            try:
                s = run(expr)
            except LispError:
                print "EVAL ERROR:"
                import traceback
                traceback.print_exc()
            except KeyboardInterrupt: break
            else:
                try:
                    print s
                except:
                    print '???'
    except (EOFError, KeyboardInterrupt):
        print "Done"


def usage():
    print """Usage:
    lisp.py            # Runs a Read-Eval-Print Loop.
    lisp.py file.pyl   # Executes file.pyl
    """


def main():
    import sys, getopt

    args_in = sys.argv[1:]
    longopts = [
        'help',
        'verbose',
    ]
    shortopts = '?hv'
    opts, args = getopt.getopt(args_in, shortopts, longopts)

    opt_help = False
    opt_verbose = 0

    for opt, value in opts:
        if opt in ['-h', '-?', '--help']:
            opt_help = True
        elif opt in ['-v', '--verbose']:
            opt_verbose += 1

    debug = (opt_verbose >= 1)

    if opt_help:
        usage()
        return
    elif not args:
        repl(debug=debug)
    else:
        # Each spare arg is interpreted as a file name
        for filename in args:
            if debug:
                print "Reading", filename
            reader = FileReader(filename)
            for expr in reader:
                result = lisp_eval(expr)

if __name__ == '__main__':
    main()
