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
        pass
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

# Maths functions
    def plus(*args):
        return sum(args)
    def times(*args):
        result = 1
        for arg in args:
            result *= arg
        return result
    def minus(l, r):
        return l - r
    def divide(l, r):
        return l / r

# IO functions
    def print_func(*args):
        for arg in args:
            print arg,
        print

# The environment, with Lisp names.
    env = LispEnv(**{
        '+': plus,
        '-': minus,
        '*': times,
        '/': divide,
        'print': print_func,
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
    _nl_expr = re.compile(r"[\\\n\\\r]")
    _ws_expr = re.compile(r"[\\\t\\\n\\\r, ]")
    _ws_char = ' '

    def __init__(self, input_func=raw_input):
        self._raw_input = input_func
        self._buffer = ''

    def _input(self):
        data = ''
        while not data:
            data = self._raw_input(self._prompt)
        return data + "\n" # This nl gets stripped by python :-(
    def _read(self):
        data = str(self._input())
        # Replace all whitespace chars with the one true ws-char.
#        data = self._ws_expr.sub(self._ws_char, data)
        self._buffer += data

    def _drop_until(self, expr, matches=True):
        while (expr.match(self.peek_char()) is not None) == matches:
            self._buffer = self._buffer[1:]
    def _drop_ws(self):
        self._drop_until(self._ws_expr)
    def _drop_until_newline(self):
        self._drop_until(self._nl_expr, False)

    def get_char(self):
        c = self.peek_char()
        c, self._buffer = self._buffer[0], self._buffer[1:]
        return c

    def peek_char(self):
        if not self._buffer:
            self._read()
        return self._buffer[0]

    def get_non_ws_char(self):
        self._drop_ws()
        return self.get_char()

    def get_expr(self,
        digits='0123456789',
        symbol_start_chars='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_+-/*!?$^~#',
        symbol_chars=''
    ):
        c = self.get_non_ws_char() # First char.
        if c == ';':
            self._drop_until_newline()
            return None
        if c == '(':
            self._drop_ws()
            s_expr = []
            while self.peek_char() != ')':
                s_expr.append(self.get_expr())
                self._drop_ws()
            # consume the ')'
            self.get_char()
            return SExpr(s_expr)
        if c in digits:
            n = c
            # TODO: Add float support
            while self.peek_char() in digits:
                n += self.get_char()
            return int(n)
        if c in symbol_start_chars:
            sym = c
            c = self.peek_char()
            while c in symbol_chars or c in symbol_start_chars:
                sym += c
                self.get_char()
                c = self.peek_char()
            return Symbol(sym)
        raise ReaderError("Unexpected symbol", c, self._buffer)

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
# The evaluator

def lisp_eval(expr, env=basic_env):
    type_e = type(expr)
    if type_e is Symbol:
        if   expr == 'nil': return None
        elif expr == 't': return True
        elif expr == 'f': return False
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
                assert len(r) == 2
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
            # TODO: implement progn, eval a sequence
            # TODO: implement fn[], define a function

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
