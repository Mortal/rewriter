import ast
import types
import inspect
import textwrap

import numpy as np


def get_source(fn):
    if isinstance(fn, (types.ModuleType, types.FunctionType, types.LambdaType)):
        lines, lineno = inspect.getsourcelines(fn)
        filename = inspect.getsourcefile(fn)
        source = ''.join(lines)
        return textwrap.dedent(source), filename, lineno
    elif isinstance(fn, str):
        return textwrap.dedent(fn), '<unknown>', 1
    else:
        raise NotImplementedError


def get_basename(node):
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Call):
        return get_basename(node.func)
    elif isinstance(node, ast.Attribute):
        return node.attr


def strip_decorators(decorator_list, fn):
    i = next(i for i, d in enumerate(decorator_list)
             if get_basename(d) == fn.__name__)
    del decorator_list[:i+1]


Toplevel = tuple(
    getattr(ast, mod) for mod in
    'Module Interactive Expression Suite'.split())

Statement = tuple(
    getattr(ast, stmt) for stmt in '''FunctionDef AsyncFunctionDef ClassDef
    Return Delete Assign AugAssign For AsyncFor While If With AsyncWith Raise
    Try Assert Import ImportFrom Global Nonlocal Expr Pass Break
    Continue'''.split())

Expr = tuple(
    getattr(ast, expr) for expr in '''BoolOp BinOp UnaryOp Lambda IfExp Dict
    Set ListComp SetComp DictComp GeneratorExp Await Yield YieldFrom Compare
    Call Num Str Bytes NameConstant Ellipsis Attribute Subscript Starred Name
    List Tuple'''.split())

LValue = tuple(
    getattr(ast, lv) for lv in 'Attribute Subscript Name'.split())


class NodeTransformer(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.node_path = []
        self.append_dest = []

    def append(self, node):
        self.append_dest[-1].append(node)

    def copy_location(self, node):
        return ast.copy_location(node, self.node_path[-1])

    def generic_visit(self, node):
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                new_values = []
                if all(isinstance(value, Statement) for value in old_value):
                    self.append_dest.append(new_values)
                for value in old_value:
                    if isinstance(value, ast.AST):
                        self.node_path.append(value)
                        value = self.visit(value)
                        self.node_path.pop()
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    new_values.append(value)
                if all(isinstance(value, Statement) for value in old_value):
                    self.append_dest.pop()
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
        return node

    def append_single(self, source):
        mod = ast.parse(source, mode='single')
        for node in mod.body:
            self.append(self.copy_location(node))

    def append_print(self, s):
        self.append_single('print(%r)' % (s,))


class Optimizer(NodeTransformer):
    def __init__(self):
        super().__init__()
        self.nonce = 0

    def fresh_var(self):
        self.nonce += 1
        return 't%03d' % self.nonce

    def visit_BinOp(self, node):
        optype = type(node.op).__name__
        methodname = 'visit_BinOp_' + optype
        method = getattr(self, methodname, super().generic_visit)
        return method(node)

    def assign(self, target, value):
        self.append_print("%s = %s" % (target, ast.dump(value)))
        self.append(self.copy_location(
            ast.Assign([ast.Name(target, ast.Store())], value)))

    def assign_copy(self, target, value):
        self.append_single('import copy')
        self.assign(target,
            ast.Call(
                ast.Attribute(ast.Name('copy', ast.Load()), 'copy', ast.Load()),
                [value], []))

    def visit_BinOp_commutative(self, node):
        name = self.fresh_var()
        if isinstance(node.left, LValue):
            if isinstance(node.right, LValue):
                self.append_print("Copy %s" % (ast.dump(node.left),))
                self.assign_copy(name, self.visit(node.left))
                right = self.visit(node.right)
            else:
                self.assign(name, self.visit(node.right))
                right = self.visit(node.left)
        else:
            self.assign(name, self.visit(node.left))
            right = self.visit(node.right)
        self.append_print("Add %s to %s" % (ast.dump(right), name))
        self.append(self.copy_location(
            ast.AugAssign(ast.Name(name, ast.Store()), node.op, right)))
        return ast.Name(name, ast.Load())

    visit_BinOp_Add = visit_BinOp_commutative
    visit_BinOp_Mult = visit_BinOp_commutative


def optimize(fn):
    source, filename, lineno = get_source(fn)
    module = ast.parse(source, filename)
    assert isinstance(module, ast.Module)
    tree, = module.body
    assert isinstance(tree, ast.FunctionDef)
    strip_decorators(tree.decorator_list, optimize)
    ast.increment_lineno(tree, lineno - 1)
    module = Optimizer().visit(module)
    ast.fix_missing_locations(module)
    res = {}
    eval(compile(module, filename, 'exec'), globals(), res)
    v, = res.values()
    return v


def main():
    @optimize
    def test(x):
        x = np.asarray(x)
        n, = x.shape
        return x * x + x

    print(test(np.asarray([1,2,3])))


if __name__ == "__main__":
    main()
