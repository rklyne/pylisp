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
class LispTraceableError(LispError):
    # TODO: Implement Lisp tracebacks
    pass
class LispRuntimeError(RuntimeError): pass
class ReaderError(LispError): pass
class ExpressionEvalError(LispError): pass
class LookupError(LispError): pass


# s-expression type
class SExpr(tuple): pass
class Sequence(tuple): pass

class Symbol(str):
    def __str__(self):
        return "<Symbol: %s>" % super(Symbol, self).__str__()

# enivronment
class LispEnv(dict):
    def __init__(self, mapping, parent=None):
        super(LispEnv, self).__init__(mapping)
        if parent is not None:
            assert isinstance(parent, LispEnv), parent
        self.parent = parent
    def __getitem__(self, key, *t, **k):
        if not self._has_key(key):
            if self.parent is not None:
                return self.parent.__getitem__(key, *t, **k)
        return super(LispEnv, self).__getitem__(key, *t, **k)
    def __setitem__(self, *t, **k):
        assert type(t[0]) is Symbol, (t[0], type(t[0]))
        return super(LispEnv, self).__setitem__(*t, **k)
    def _has_key(self, key):
        return super(LispEnv, self).has_key(key)
    def has_key(self, key):
        if self._has_key(key):
            return True
        return self.parent.has_key(key)

class LispFunc(object):
    extra_bindings = None
    """In general, a Python function is a PyLisp function, and a PyLisp function
    needs this wrapper to make it
    """
    def __init__(self, bindings, exprs, env):
        assert isinstance(env, LispEnv), env
        assert isinstance(bindings, Sequence), bindings
        self.exprs = exprs
        exact_bindings = bindings
        if bindings[-1][0] == '&':
            self.extra_bindings = Symbol(bindings[-1][1:])
            exact_bindings = bindings[:-1]
        self.bindings = exact_bindings
        self.def_env = env
    def __call__(self, *args_in, **kwargs_in):
        # Implemented this so that Lisp code is Python callable.
        # (Python code is Lisp callable, so this is also required to make Lisp
        # functions Lisp callable ;))
        ret = None
        local_env = {}
        args = args_in
        for name in self.bindings:
            assert isinstance(name, Symbol), (name, self.bindings)
            local_env[name] = args[0]
            args = args[1:]
        if self.extra_bindings:
            local_env[self.extra_bindings] = args
        elif args or kwargs_in:
            raise LispError("Too many params to function", self, args_in)
        env = LispEnv(local_env, parent=self.def_env)
        for expr in self.exprs:
            ret = lisp_eval(expr, env)
        return ret

class LispMacro(LispFunc):
    _lisp_macro = True

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
        for arg in args:            res += str(arg)
        return res

# Sequence functions
    def seq(*args):
        return SExpr(args)
    def head(seq):
        return seq[0]
    def tail(seq):
        return seq[1:]

    def cons(el, seq):
        return Sequence((el,)) + seq

# IO functions
    def print_func(*args):
        for arg in args:
            print arg,
        print

# System functions
    def assert_func(truth, message=_no_value):
        if message is _no_value:
            assert truth
        else:
            assert truth, message

    def _lisp_eval(code):
        return lisp_eval(code)

# Boolean logic functions
    def equals(a, b):
        return a == b

# The environment, with Lisp names.
    env = LispEnv({
        't': True,
        '+': plus,
        '-': minus,
        '*': times,
        '/': divide,
        'print': print_func,
        'str': string,
        'assert': assert_func,
        'apply': apply,
        'list': seq,
        'seq': seq,
        'head': head,
        'tail': tail,
        'cons': cons,
        'eq': equals,
        'eval': _lisp_eval,
    })

    return env

# basic_env is the bare minimum Lisp environment
_basic_env = build_basic_env()
basic_env = None
# simple_env is an extension os basic_env with core.pyl loaded
_simple_env = None
simple_env = None
def reset():
    global basic_env
    global _simple_env
    global simple_env
    basic_env = None
    _simple_env = None
    simple_env = None
reset()

def get_basic_env():
    global basic_env
    if basic_env is None:
        basic_env = LispEnv({}, _basic_env)
    return basic_env

def build_simple_env():
    env = LispEnv({}, get_basic_env())
    core_file_name = 'core.pyl'
    core_reader = FileReader(core_file_name)
    for expr in core_reader:
        lisp_eval(expr, env)
    return env
def get_simple_env():
    global _simple_env
    global simple_env
    if simple_env is None:
        if _simple_env is None:
            _simple_env = build_simple_env()
        simple_env = LispEnv({}, _simple_env)
    return simple_env

def resolve_def(var, env1, env2):
    try:
        return env1[var]
    except KeyError:
        try:
            return env2[var]
        except KeyError:
            raise LookupError(var)

# The Reader

class Reader(object):
    """The default Reader implementation. Reads input with Python's `raw_input`
    method.
    """
    _prompt = 'Lisp --> '
    _nl_expr = re.compile(r"[\n\r]")
    _ws_expr = re.compile(r"[\t\n\r, ]")
    _debug = True
    symbol_start_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_+-/*!?$&^~#'
    digits = '0123456789'
    symbol_chars = digits

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

    def get_expr(self):
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
        # 0.1 '-quoting.
        if c == "'":
            quoted_expr = self.get_expr()
            return SExpr((Symbol('quote'), quoted_expr))
        # 0.2 #-macros
        if c == '#':
            raise ReaderError, "No such macro"
        # 1. A Symbol
        if c in self.symbol_start_chars:
            sym = c
            c = self.peek_char()
            while c and (c in self.symbol_chars or c in self.symbol_start_chars):
                sym += c
                self.get_char()
                c = self.peek_char()
            return Symbol(sym)
        # 2. A number
        if c in self.digits:
            n = c
            while self.peek_char() and self.peek_char() in self.digits:
                n += self.get_char()
            # Detect floating point numbers:
            if self.peek_char() == '.':
                n += self.get_char()
                while self.peek_char() and self.peek_char() in self.digits:
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
def lisp_eval(expr, env=None, private_env=None):
    """PyLisp Evaluator.
    Reserved words:
    * def
    * defmacro
    * progn
    * quote
    * nil
    * t
    * if
    * fn
    """
    if env is None:
        env = get_simple_env()
    local_env = LispEnv({}, private_env)

    type_e = type(expr)
    if type_e is Symbol:
        # 'nil' and 't' can't be overridden.
        if   expr == 'nil': return None
        elif expr == 't': return True
        else: return resolve_def(expr, local_env, env)
    elif type_e is int: return expr
    elif type_e is float: return expr
    elif isinstance(expr, tuple):
        if len(expr) == 0:
            return None
        f, r = expr[0], expr[1:]
        type_f = type(f)
        if f == 'def':
            assert len(r) == 2, r
            name = r[0]
            assert type(name) is Symbol, name
            expr = r[1]
            value = lisp_eval(expr, env, private_env)
            env[name] = value
            return value
        elif f == 'defmacro':
            assert len(r) >= 3, r
            name = r[0]
            # A new LispMacro, with bindings and an expression body
            bindings = r[1]
            exprs = r[2:]
            value = LispMacro(bindings, exprs, env)
            env[name] = value
            value.name = name
            return value
        elif f == 'progn':
            ret = None
            while r:
                ret = lisp_eval(r[0], env, private_env)
                r = r[1:]
            return ret
        elif f == 'quote':
            assert len(r) == 1, expr
            return r[0]
        elif f == 'fn':
            assert len(r) >= 2, r
            # A new LispFunc, with bindings and an expression body
            return LispFunc(r[0], r[1:], env)
        elif f == 'if':
            else_body = None
            if len(r) == 3:
                else_body = r[2]
            test = r[0]
            body = r[1]
            if lisp_eval(test, env, private_env):
                return lisp_eval(body, env, private_env)
            elif else_body is not None:
                return lisp_eval(else_body, env, private_env)
        elif f == 'let':
            # TODO: implement 'let'
            raise NotImplementedError
        elif f == 'binding':
            # TODO: implement 'binding'
            raise NotImplementedError
        else:
            func = lisp_eval(f, env, private_env)
            args = r
            macro = hasattr(func, '_lisp_macro')
            if not macro:
                args = map(lambda x: lisp_eval(x, env, private_env), args)
            ret = lisp_apply(func, args, env)
            if macro:
                return lisp_eval(ret, env, private_env)
            return ret
    else:
        raise LispRuntimeError("Cannot evaluate this expression.", expr)

run = lisp_eval

def repl(debug=False, env=None):
    """REPL:
    * R ead s-expressions and
    * E valuate them, then
    * P rint the results.
    * L oop."""
    reader = Reader()
    if env is None:
        env = get_default_env()
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
                s = lisp_eval(expr, env)
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
        'bare',
    ]
    shortopts = '?hv'
    opts, args = getopt.getopt(args_in, shortopts, longopts)

    opt_help = False
    opt_verbose = 0
    opt_bare = False

    for opt, value in opts:
        if opt in ['-h', '-?', '--help']:
            opt_help = True
        elif opt in ['-v', '--verbose']:
            opt_verbose += 1
        elif opt in ['--bare']:
            opt_bare = True

    debug = (opt_verbose >= 1)

    if opt_help:
        usage()
        return

    if opt_bare:
        env = get_basic_env()
    else:
        env = get_simple_env()

    if not args:
        repl(debug=debug, env=env)
    else:
        # Each spare arg is interpreted as a file name
        for filename in args:
            if debug:
                print "Reading", filename
            reader = FileReader(filename)
            for expr in reader:
                result = lisp_eval(expr, env=env)

if __name__ == '__main__':
    main()

