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
class LispCallError(LispError): pass
class LispRuntimeError(RuntimeError): pass
class ReaderError(LispError): pass
class ExpressionEvalError(LispError): pass
class LookupError(LispError): pass


# s-expression type
# TODO: Make s-expressions look like linked lists
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
    def __nonzero__(self):
        if super(LispEnv, self).__nonzero__():
            return True
        if self.parent is not None:
            return bool(self.parent)
        return False
    def _has_key(self, key):
        return super(LispEnv, self).has_key(key)
    def has_key(self, key):
        if self._has_key(key):
            return True
        if self.parent is not None:
            return self.parent.has_key(key)
        return False
    def lookup(self, key):
        return self[key]

    def _flatten(self):
        "Written to help debugging."
        dct = {}
        for k, v in self.items():
            dct[k] = v
        if self.parent is not None:
            dct.update(self.parent._flatten())
        return dct

class StackableLispEnv(object):
    def __init__(self, base):
        assert isinstance(base, LispEnv), base
        self._stack = []
        self._base = base
        self.push(base)

    def __str__(self):
        return '<%s "%s">' % (
            self.__class__.__name__,
            getattr(self, 'name', '(unnamed)'),
        )
    __repr__ = __str__

    def __getattr__(self, key, *t, **k):
        if key in [
            'push',
            'pop',
            '_stack',
            'define',
        ]:
            return self.__dict__[key]
        return getattr(self._stack[-1], key, *t, **k)

    def __hasattr__(self, key, *t, **k):
        return hasattr(self._stack[-1], key, *t, **k)

    def push(self, env):
        self._stack.append(env)
    def pop(self):
        self._stack.pop()
        assert self._stack

    def define(self, key, value):
        self._stack[-1][key] = value

    def lookup(self, key):
        return self._stack[-1][key]

    def new_scope(self, mapping=None):
        if mapping is None:
            mapping = {}
        return LispEnv(mapping, self._stack[-1])

class LispFunc(object):
    extra_bindings = None
    """In general, a Python function is a PyLisp function, and a PyLisp function
    needs this wrapper to make it
    """
    def __init__(self, bindings, exprs, env, local_env):
        assert isinstance(env, StackableLispEnv), env
        assert type(bindings) is Sequence, bindings
        self.expr = SExpr((Symbol("progn"), ) + exprs)
        exact_bindings = bindings
        if bindings and bindings[-1][0] == '&':
            self.extra_bindings = Symbol(bindings[-1][1:])
            exact_bindings = bindings[:-1]
        self.bindings = exact_bindings
        self.def_env = env
        self.def_local_env = LispEnv({}, local_env)
    def __call__(self, *args_in, **kwargs_in):
        # Implemented this so that Lisp code is Python callable.
        # (Python code is Lisp callable, so this is also required to make Lisp
        # functions Lisp callable ;))
        ret = None
        args = args_in
        for name in self.bindings:
            assert isinstance(name, Symbol), (name, self.bindings)
            self.def_local_env[name] = args[0]
            args = args[1:]
        if self.extra_bindings:
            self.def_local_env[self.extra_bindings] = args
        elif args or kwargs_in:
            raise LispError("Too many params to function", self, args_in, kwargs_in)
        env = get_thread_env()
        if self.def_env is not env:
            raise LispCallError("Data from another Lisp", env)
        ret = lisp_eval(self.expr, env, self.def_local_env)
        return ret

_thread_envs = {}
def get_thread_env():
    import thread
    thread_id = thread.get_ident()
    env = _thread_envs.get(thread_id)
    if env is not None:
        return env
    return None
def set_thread_env(env):
    import thread
    thread_id = thread.get_ident()
    _thread_envs[thread_id] = env

class LispMacro(LispFunc):
    _lisp_macro = True

env_stack = []
def lisp_apply(f, args, env):
    assert callable(f), f
    try:
        # XXX: Not thread safe except in pure PyLisp code
        env0 = get_thread_env()
        set_thread_env(env)
        return f(*args)
    finally:
        set_thread_env(env0)

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
    def list_(*args):
        return SExpr(args)
    def seq(*args):
        return Sequence(args)
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

    def lisp_eval_(code):
        return lisp_eval(code)

# Boolean logic functions
    def equals(a, b):
        return a == b
    def not_(a):
        return not a

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
        'list': list_,
        'seq': seq,
        'head': head,
        'tail': tail,
        'cons': cons,
        'eq': equals,
        '=': equals,
        'not': not_,
        'eval': lisp_eval_,
    })

    return env

# basic_env is the bare minimum Lisp environment
_basic_env = build_basic_env()
basic_env = None
# simple_env is an extension os basic_env with core.pyl loaded
_simple_env = None
def reset():
    global basic_env
    global _simple_env
    basic_env = None
    _simple_env = None
    _thread_envs.clear()
    env_stack[:] = []
reset()

def make_stackable_env(env, name='(no name)'):
    new_env = StackableLispEnv(env)
    new_env.name = name
    return new_env

def _get_basic_env():
    global basic_env
    if basic_env is None:
        basic_env = LispEnv({}, _basic_env)
    return basic_env
def get_basic_env():
    return make_stackable_env(_get_basic_env(), 'basic')

def build_simple_env():
    base_env = LispEnv({}, _get_basic_env())
    env = make_stackable_env(base_env, 'simple')
    core_file_name = 'core.pyl'
    core_reader = FileReader(core_file_name)
    exprs = []
    for expr in core_reader:
        exprs.append(expr)
    expr = SExpr((Symbol("progn"), SExpr(exprs)))
    lisp_eval(expr, env, LispEnv({}))
    return env
def get_simple_env():
    global _simple_env
    if _simple_env is None:
        _simple_env = build_simple_env()
    return _simple_env

def resolve_def(var, env1, env2):
    try:
        return env1.lookup(var)
    except KeyError:
        try:
            return env2.lookup(var)
        except KeyError:
            raise LookupError(var)

# The Reader

class UnquoteWrapper(object):
    def __init__(self, content):
        self.content = content

def normality_(x):
    assert type(x) is not UnquoteWrapper, x
    return x

class Reader(object):
    """The default Reader implementation. Reads input with Python's `raw_input`
    method.
    """
    _prompt = 'Lisp --> '
    _nl_expr = re.compile(r"[\n\r]")
    _ws_expr = re.compile(r"[\t\n\r, ]")
    _debug = True
    symbol_start_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_+-/*!?$&^#='
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
        # TODO: This could be much more efficient if I dropped several chars at
        # once. The RE is matching them all anyway...
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
        """Ensure that there is data, excepting at EOF.
        Returns one character without consuming it.
        """
        self._ensure_data()
        if not self._buffer:
            return ""
        return self._buffer[0]

    def get_non_ws_char(self):
        self._drop_ws()
        return self.get_char()

    def get_expr(self, return_func=normality_):
        """Reads a form and returns one full s-expression. An s-expression may be:
        0. A reader macro. These produce more complex lisp forms.
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
            quoted_expr = self.get_expr(return_func=return_func)
            return return_func(SExpr((Symbol('quote'), quoted_expr)))
        # 0.2.a `()-quoting. (backtick)
        if c == "`":
            # TODO: Implement this quoting mechanism:
            # In "`(+ x 1 ~a)" everything is quoted but 'a'.
            def quote_(expr):
                if type(expr) is UnquoteWrapper:
                    return expr.content
                else:
                    return expr
            return self.get_expr(return_func=quote_)
        # 0.2.b `,-quoting's ~unquote mechanism
        if c == "~":
            return return_func(UnquoteWrapper(self.get_expr(return_func=normality_)))
        # 0.3 #-macros
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
            return return_func(Symbol(sym))
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
                return return_func(float(n))
            return return_func(int(n))
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
            return return_func(s)
        # 4. Comment
        if c == ';':
            self._drop_until_newline()
            return return_func(None)
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
                s_expr.append(self.get_expr(return_func=return_func))
                self._drop_ws()
            # consume the closing char
            self.get_char()
            return return_func(expr_class(s_expr))
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
# TODO: Build a dictionary of symbols to lambdas to handle special forms quickly.
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
    * let
    * binding
    """
    if env is None:
        env = get_simple_env()
    assert type(env) is StackableLispEnv, (type(env), env)
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
            # Lisp "()" == Lisp "nil"
            return None
        f, r = expr[0], expr[1:]
        if f == 'def':
            assert len(r) == 2, r
            name = r[0]
            assert type(name) is Symbol, name
            expr = r[1]
            value = lisp_eval(expr, env, private_env)
            env.define(name, value)
            assert env.has_key(name), env._flatten()
            return value
        elif f == 'defmacro':
            assert len(r) >= 3, r
            name = r[0]
            # A new LispMacro, with bindings and an expression body
            bindings = r[1]
            exprs = r[2:]
            value = LispMacro(bindings, exprs, env, private_env)
            env.define(name, value)
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
            return LispFunc(r[0], r[1:], env, private_env)
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
            bindings = r[0]
            body = r[1:]
            assert isinstance(bindings, Sequence)
            new_private_env = LispEnv({}, private_env)
            while bindings:
                name = bindings[0]
                expr = bindings[1]
                new_private_env[name] = lisp_eval(expr, env, new_private_env)
                bindings = bindings[2:]
            body = SExpr((Symbol("progn"), ) + body)
            return lisp_eval(body, env, new_private_env)
        elif f == 'binding*':
            bindings = r[0]
            body = r[1:]
            assert isinstance(bindings, Sequence)
            new_bindings = {}
            while bindings:
                name = bindings[0]
                assert type(name) is Symbol, name
                expr = bindings[1]
                new_bindings[name] = lisp_eval(expr, env, private_env)
                bindings = bindings[2:]
            new_env = env.new_scope(new_bindings)
            try:
                env.push(new_env)
                return lisp_eval(body, env, private_env)
            finally:
                env.pop()
        else:
            # If not a special form, then treat this as a function.
            # Evaluate the first form and keep it as the function object.
            func = lisp_eval(f, env, private_env)
            assert callable(func), (func, f, expr)
            args = r
            macro = hasattr(func, '_lisp_macro')
            if not macro:
                args = map(lambda x: lisp_eval(x, env, private_env), args)
                ret = lisp_apply(func, args, env)
                return ret
            else:
                ret = lisp_apply(func, args, env)
                return lisp_eval(ret, env, private_env)
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
        while True:
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

