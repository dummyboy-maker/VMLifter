from collections import defaultdict

from toolsFunc import debug


class VarType:
    IMM = 0
    REG = 1
    MEM = 2
    VMSTACK = 3


class VarName:
    CONST = "const"


class Var:
    def __init__(self, name: str, type: int, value: int):
        self.name = name
        self.type = type
        self.value = value

    def __repr__(self):
        if self.type == VarType.IMM:
            # if self.type == VarType.IMM or self.type == VarType.VMSTACK:
            return hex(self.value)
        else:
            return self.name

    def __eq__(self, other):
        if isinstance(other, Var):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)


class Expr:
    def __init__(self, op: str, left, right):
        self.op = op  # op type：+, -, *, /
        self.left = left  # left operand（Expr or var）
        self.right = right  # right operand（Expr or var）

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


class SliceExprState:
    def __init__(self):
        self.vars_cnt = defaultdict(int)  # var_name: current cnt
        self.exprs = {}  # var: expr

    def isValueStable(self, var_name: str, var_value: int):
        for var in self.exprs:
            # 值依赖缺失
            if var.name == var_name and var.value != var_value:
                return False
        return True

    def createUseVar(self, var_name: str, var_type: int, var_value: int):

        if var_name == VarName.CONST:
            return Var(var_name, var_type, var_value)

        var_cnt = self.vars_cnt.get(var_name, 0)
        var_name_with_cnt = '%s_%d' % (var_name, var_cnt)

        # 先查找是否有此变量的定义
        # 因为污点分析后的指令是不连续的, 存在值依赖缺失的问题
        # 如果没有找到此变量的定义，则定义此变量，否则返回已定义的变量
        if not self.isValueStable(var_name_with_cnt, var_value):
            var_cnt += 1
            var_name_with_cnt = '%s_%d' % (var_name, var_cnt)
            self.vars_cnt[var_name] = var_cnt

        return Var(var_name_with_cnt, var_type, var_value)

    def createDefVar(self, var_name: str, var_type: int, var_value: int):

        var_cnt = self.vars_cnt.get(var_name, 0) + 1
        var_name_with_cnt = '%s_%d' % (var_name, var_cnt)
        self.vars_cnt[var_name] = var_cnt

        return Var(var_name_with_cnt, var_type, var_value)

    def backTraceExpr(self, node):
        # 对value进行回溯
        if isinstance(node, Var):
            # 立即数、内存、寄存器
            # 立即数直接返回值
            # 内存和寄存器则寻找表达式，找不到则返回对应汇编指令处的值，否则继续递归回溯

            if node.type == VarType.IMM:
                return node
            elif node.type == VarType.REG or node.type == VarType.VMSTACK:
                # reg类型一定能回溯到mem vmstack
                # vmstack可能回溯到mem reg，甚至找不到值依赖
                # 每次Var的时候，产生的对象是不同的(地址不同), 要想能够查找到，重写Var的__eq__和__hash__
                expr = self.exprs.get(node)
                debug("\tfind %s in exprs: %s" % (node, expr))
                if expr:
                    return self.backTraceExpr(expr)
                else:
                    # vmstack 回溯不到值依赖, 返回具体值，封装成IMM
                    # return node
                    return Var(VarName.CONST, VarType.IMM, node.value)
            elif node.type == VarType.MEM:
                # 对于真实内存，终止回溯
                # 其一是已经确定到了真实内存中的数据
                # 其二是如果继续回溯，会造成堆栈溢出
                return node

        elif isinstance(node, Expr):
            l = self.backTraceExpr(node.left)
            r = self.backTraceExpr(node.right)
            return Expr(node.op, l, r)
        else:
            return node

    def concretizeExprValue(self, node) -> str:
        if isinstance(node, Var):  # only VarType.MEM
            return hex(node.value)
        elif isinstance(node, Expr):
            l = self.concretizeExprValue(node.left)
            r = self.concretizeExprValue(node.right)
            return l + ' ' + node.op + ' ' + r


def isContainMem(node):
    # 情况1：变量
    if isinstance(node, Var):
        return node.type == VarType.MEM

    # 情况2：表达式
    elif isinstance(node, Expr):
        return isContainMem(node.left) or isContainMem(node.right)

    return False