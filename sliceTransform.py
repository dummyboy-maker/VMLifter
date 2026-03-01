from typing import List

from triton import *

from opWrapper import *
from toolsFunc import debug
from unidbgTraceParser import InstructionInfo

ctx = TritonContext(ARCH.AARCH64)
vm_stack_addr = 0



"""谨记 先定义 use 变量，最后定义 def 变量"""

SHIFT_MAP = {
    AST_NODE.BVSHL: "lsl",
    AST_NODE.BVLSHR: "lsh",
    AST_NODE.BVASHR: "asr",
    AST_NODE.BVROR: "ror"
}


def handleADD(inst_info: InstructionInfo, ses: SliceExprState):
    # add x7, x23, x7           支持
    # add w0, w1, #5            支持
    # add x0, x1, x2, lsl #3    未测试
    # add x0, x1 等价于 add x0, x0, x1 支持
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))
        # 需要事先符号化寄存器变量，否则ast解析时获取不到寄存器
        ctx.symbolizeRegister(getattr(ctx.registers, reg))
    # debug(f"symbol register: {ctx.getSymbolicRegisters()}")
    ctx.processing(inst)
    # debug(f"instruction symbolic exprs: {inst.getSymbolicExpressions()}")

    dest_reg = inst.getOperands()[0]
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()

    # 根据源操作数创建变量
    src_vars = []
    for reg, expr in inst.getReadRegisters():
        # expr 是 ast
        # debug(f"expr: {type(expr)}")
        expr_type = expr.getType()
        if expr_type == AST_NODE.BVSHL:     # 左移
            reg_node, shift_node = expr.getChildren()
            reg_name = str(reg_node).split('_')[0]
            reg_value = int(regs_before.get(reg_name), 16)
            use_var1 = ses.createUseVar(reg_name, VarType.REG, reg_value)
            use_var2 = ses.createUseVar(VarName.CONST, VarType.IMM, int(str(shift_node), 16))
            expr = Expr("<<", use_var1, use_var2)
            src_vars.append(expr)
        elif expr_type in SHIFT_MAP:
            raise Exception("Unsupport shift type in ADD instruction")
        else:
            reg_name = reg.getName()
            reg_value = int(regs_before.get(reg_name), 16)
            use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
            src_vars.append(use_var)

    # 也许存在操作数, 例如 add w0, w1, #5
    for imm, expr in inst.getReadImmediates():
        use_var = ses.createUseVar(VarName.CONST, VarType.IMM, imm.getValue())
        src_vars.append(use_var)

    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    right_expr = Expr('+', src_vars[0], src_vars[1])

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleSUB(inst_info, ses):
    # "sub x25, x25, x22" x25=0x0 x22=0x0 => x25=0x0
    # "sub sp, sp, #0xc0" sp=0xe4fff5f0 => sp=0xe4fff530
    # SUB <Wd|Xd>, <Wn|Xn>, <Wm|Xm>, <shift> #<amount>
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))
        # 需要事先符号化寄存器变量，否则ast解析时获取不到寄存器
        ctx.symbolizeRegister(getattr(ctx.registers, reg))
    # debug(f"symbol register: {ctx.getSymbolicRegisters()}")
    ctx.processing(inst)
    # debug(f"instruction symbolic exprs: {inst.getSymbolicExpressions()}")

    dest_reg = inst.getOperands()[0]
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()

    # 根据源操作数创建变量
    src_vars = []
    for reg, expr in inst.getReadRegisters():
        # expr 是 ast
        # debug(f"expr: {type(expr)}")
        expr_type = expr.getType()
        if expr_type == AST_NODE.BVSHL:  # 左移
            reg_node, shift_node = expr.getChildren()
            reg_name = str(reg_node).split('_')[0]
            reg_value = int(regs_before.get(reg_name), 16)
            use_var1 = ses.createUseVar(reg_name, VarType.REG, reg_value)
            use_var2 = ses.createUseVar(VarName.CONST, VarType.IMM, int(str(shift_node), 16))
            expr = Expr("<<", use_var1, use_var2)
            src_vars.append(expr)
        elif expr_type in SHIFT_MAP:
            raise Exception("Unsupport shift type in SUB instruction")
        else:
            reg_name = reg.getName()
            reg_value = int(regs_before.get(reg_name), 16)
            use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
            src_vars.append(use_var)

    # 也许存在操作数, 例如 add w0, w1, #5
    for imm, expr in inst.getReadImmediates():
        use_var = ses.createUseVar(VarName.CONST, VarType.IMM, imm.getValue())
        src_vars.append(use_var)

    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    right_expr = Expr('-', src_vars[0], src_vars[1])

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleLDR(inst_info: InstructionInfo, ses: SliceExprState):
    # ldr x1, [x2, #9]  => x1 = mem[x2+9] or x1 = vmstack[x2+9]
    # return x1, mem[x2+9] or x1, vmstack[x2+9]

    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before
    regs_after = inst_info.regs_after

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)
    # debug(inst.getSymbolicExpressions())

    oprands = inst.getOperands()
    dest_reg = oprands[0]
    mem_op = oprands[1]

    dest_reg_name = dest_reg.getName()
    var_value = int(regs_after.get(dest_reg_name, '0x0'), 16)

    # 可能存在类似指令:"ldrb w7, [x7]" x7=0x12709000 => w7=0x2f,
    # 该指令使用同一个寄存器，所以基址直接从trace中提取
    base_reg_name = mem_op.getBaseRegister().getName()
    base_addr = int(regs_before.get(base_reg_name), 16)
    mem_addr = mem_op.getAddress()

    if base_addr == vm_stack_addr:  # 内存值来自虚拟栈
        var_name = 'vmstack[0x%x]' % mem_addr
        use_var = ses.createUseVar(var_name, VarType.VMSTACK, var_value)
    else:
        var_name = 'mem[0x%x]' % mem_addr
        use_var = ses.createUseVar(var_name, VarType.MEM, var_value)

    # 谨记先 use 后 def
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, var_value)

    debug(f'{def_var.name} = {use_var.name}')

    ses.exprs.update({def_var: use_var})


def handleSTR(inst_info, ses):
    # "str x7, [x0, #0x90]" x7=0x12709000 x0=0x127090c0 => x7=0x12709000
    # mem[x0+0x90] = x7 or vmstack[x0+0x90] = x7

    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)
    # debug(inst.getSymbolicExpressions())

    oprands = inst.getOperands()
    src_reg = oprands[0]
    mem_op = oprands[1]

    src_reg_name = src_reg.getName()
    var_value = ctx.getConcreteRegisterValue(src_reg)
    use_var = ses.createUseVar(src_reg_name, VarType.REG, var_value)

    base_reg = mem_op.getBaseRegister()
    base_addr = ctx.getConcreteRegisterValue(base_reg)
    mem_addr = mem_op.getAddress()

    if base_addr == vm_stack_addr:  # 内存值来自虚拟栈
        var_name = 'vmstack[0x%x]' % mem_addr
        def_var = ses.createDefVar(var_name, VarType.VMSTACK, var_value)
    else:
        var_name = 'mem[0x%x]' % mem_addr
        def_var = ses.createDefVar(var_name, VarType.MEM, var_value)

    debug(f'{def_var.name} = {use_var.name}')

    ses.exprs.update({def_var: use_var})


def handleLDP(inst_info, ses):
    # ldp x1, x0, [x9]
    # ldp w7, w23, [x0, #0x70]  => w7_1 = vmstack[0x127090c0 + 0x70]  w23_2 = vmstack[0x127090c0 + 0x74]

    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before
    regs_after = inst_info.regs_after

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)
    # debug(inst.getSymbolicExpressions())

    # 解析操作数
    oprands = inst.getOperands()
    dest_reg1 = oprands[0]
    dest_reg2 = oprands[1]
    mem_op = oprands[2]

    dest_reg1_name = dest_reg1.getName()
    dest_reg2_name = dest_reg2.getName()
    dest_reg1_value = int(regs_after.get(dest_reg1_name), 16)
    dest_reg2_value = int(regs_after.get(dest_reg2_name), 16)

    reg_size = mem_op.getSize() // 2
    mem_addr = mem_op.getAddress()
    base_reg_name = mem_op.getBaseRegister().getName()
    base_addr = int(regs_before.get(base_reg_name), 16)

    if base_addr == vm_stack_addr:  # 内存值来自虚拟栈
        var1_name = 'vmstack[0x%x]' % mem_addr
        var2_name = 'vmstack[0x%x]' % (mem_addr + reg_size)
        use_var1 = ses.createUseVar(var1_name, VarType.VMSTACK, dest_reg1_value)
        use_var2 = ses.createUseVar(var2_name, VarType.VMSTACK, dest_reg2_value)
    else:
        var1_name = 'mem[0x%x]' % mem_addr
        var2_name = 'mem[0x%x]' % (mem_addr + reg_size)
        use_var1 = ses.createUseVar(var1_name, VarType.MEM, dest_reg1_value)
        use_var2 = ses.createUseVar(var2_name, VarType.MEM, dest_reg2_value)

    def_var1 = ses.createDefVar(dest_reg1_name, VarType.REG, dest_reg1_value)
    def_var2 = ses.createDefVar(dest_reg2_name, VarType.REG, dest_reg2_value)

    debug(f'{def_var1.name} = {use_var1.name}')
    debug(f'{def_var2.name} = {use_var2.name}')

    ses.exprs.update({def_var1: use_var1, def_var2: use_var2})


def handleSTP(inst_info, ses):
    # stp xzr, x22, [x19, #0x58]
    # mem = reg
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    # 解析操作数
    oprands = inst.getOperands()
    src_reg1 = oprands[0]
    src_reg2 = oprands[1]
    mem_op = oprands[2]

    src_reg1_name = src_reg1.getName()
    src_reg2_name = src_reg2.getName()
    src_reg1_value = ctx.getConcreteRegisterValue(src_reg1)
    src_reg2_value = ctx.getConcreteRegisterValue(src_reg2)

    # create use
    use_var1 = ses.createUseVar(src_reg1_name, VarType.REG, src_reg1_value)
    use_var2 = ses.createUseVar(src_reg2_name, VarType.REG, src_reg2_value)

    reg_size = mem_op.getSize() // 2
    mem_addr = mem_op.getAddress()
    base_reg = mem_op.getBaseRegister()
    base_addr = ctx.getConcreteRegisterValue(base_reg)

    # create def
    if base_addr == vm_stack_addr:  # 内存值来自虚拟栈
        var1_name = 'vmstack[0x%x]' % mem_addr
        var2_name = 'vmstack[0x%x]' % (mem_addr + reg_size)
        def_var1 = ses.createDefVar(var1_name, VarType.VMSTACK, src_reg1_value)
        def_var2 = ses.createDefVar(var2_name, VarType.VMSTACK, src_reg2_value)
    else:
        var1_name = 'mem[0x%x]' % mem_addr
        var2_name = 'mem[0x%x]' % (mem_addr + reg_size)
        def_var1 = ses.createDefVar(var1_name, VarType.MEM, src_reg1_value)
        def_var2 = ses.createDefVar(var2_name, VarType.MEM, src_reg2_value)

    debug(f'{def_var1.name} = {use_var1.name}')
    debug(f'{def_var2.name} = {use_var2.name}')

    ses.exprs.update({def_var1: use_var1, def_var2: use_var2})


def handleMUL(inst_info, ses):
    # "mul w7, w7, w23" => w7_2 = w7_1 * w23_2
    # MUL 不支持立即数格式

    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    # 根据源操作数创建变量
    src_vars = []
    for reg, expr in inst.getReadRegisters():
        reg_name = reg.getName()
        reg_value = int(regs_before.get(reg_name), 16)
        use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
        src_vars.append(use_var)

    right_expr = Expr('*', src_vars[0], src_vars[1])

    dest_reg = inst.getOperands()[0]
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleDIV(inst_info, ses):
    # "div w7, w7, w23" => w7_2 = w7_1 / w23_2
    #  div不支持立即数格式

    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    # 根据源操作数创建变量
    src_vars = []
    for reg, expr in inst.getReadRegisters():
        reg_name = reg.getName()
        reg_value = int(regs_before.get(reg_name), 16)
        use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
        src_vars.append(use_var)

    right_expr = Expr('/', src_vars[0], src_vars[1])

    dest_reg = inst.getOperands()[0]
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleMSUB(inst_info, ses):
    # msub w8, w10, w9, w8
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    # 根据源操作数创建变量
    src_vars = []
    for reg, expr in inst.getReadRegisters():# getReadRegisters获取到的寄存器序列是反向的
        reg_name = reg.getName()
        reg_value = int(regs_before.get(reg_name), 16)
        use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
        src_vars.append(use_var)

    right_expr = Expr('-', src_vars[0],
                      Expr('*', src_vars[1], src_vars[2]))

    dest_reg = inst.getOperands()[0]
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleMOV(inst_info, ses):
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    ops = inst.getOperands()
    dest_reg = ops[0]
    src = ops[1]

    dest_reg_name = dest_reg.getName()
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)

    if src.getType() == OPERAND.REG:
        src_reg_name = src.getName()
        # 没必要，因为wzr找不到定义, 最后表达式回溯时就变成了具体值
        # if src_reg_name in ['wzr']
        use_var = ses.createUseVar(src_reg_name, VarType.REG, dest_reg_value)
    else:   # 立即数
        use_var = ses.createUseVar(VarName.CONST, VarType.IMM, dest_reg_value)

    # 目标操作数
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    debug(f'{def_var.name} = {use_var}')

    ses.exprs.update({def_var: use_var})


def handleMOVK(inst_info, ses):
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)
    # debug(inst.getSymbolicExpressions())

    ops = inst.getOperands()
    dest_reg = ops[0]
    imm = ops[1]  # 必定是立即数

    dest_reg_name = dest_reg.getName()
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    src_reg_value = int(regs_before.get(dest_reg_name), 16)
    # 由于带有位移操作，不好正向获取操作数位移后的结果，但是 结果 ^ 源寄存器值 = 立即数位移结果
    imm_after_shift = src_reg_value ^ dest_reg_value
    # 立即数
    use_var1 = ses.createUseVar(VarName.CONST, VarType.IMM, imm_after_shift)
    use_var2 = ses.createUseVar(dest_reg_name, VarType.REG, src_reg_value)
    # 目标操作数
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    expr = Expr('&', use_var2, use_var1)

    debug(f'{def_var.name} = {expr}')

    ses.exprs.update({def_var: expr})


def handleADRP(inst_info, ses):
    # "adrp x8, #0x122e7000" => x8 = 0x122e7000
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)
    # debug(inst.getSymbolicExpressions())

    ops = inst.getOperands()
    dest_reg = ops[0]
    src = ops[1]

    dest_reg_name = dest_reg.getName()
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)

    if src.getType() == OPERAND.REG:
        src_reg_name = src.getName()
        use_var = ses.createUseVar(src_reg_name, VarType.REG, dest_reg_value)
    else:  # 立即数
        use_var = ses.createUseVar(VarName.CONST, VarType.IMM, dest_reg_value)

    # 目标操作数
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    debug(f'{def_var.name} = {use_var}')

    ses.exprs.update({def_var: use_var})


def handleSXTH(inst_info, ses):
    # 忽略符号扩展的影响
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    ops = inst.getOperands()
    dest_reg = ops[0]
    src_reg = ops[1]

    dest_reg_name = dest_reg.getName()
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    src_reg_name = src_reg.getName()
    src_reg_value = int(regs_before.get(src_reg_name), 16)

    use_var = ses.createUseVar(src_reg_name, VarType.REG, src_reg_value)
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    debug(f'{def_var.name} = {use_var.name}')

    ses.exprs.update({def_var: use_var})


def handleCSEL(inst_info, ses):
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    ops = inst.getOperands()
    dest_reg = ops[0]
    src_reg1 = ops[1]
    src_reg2 = ops[2]

    dest_reg_name = dest_reg.getName()
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    src_reg1_name = src_reg1.getName()
    src_reg1_value = int(regs_before.get(src_reg1_name), 16)
    src_reg2_name = src_reg2.getName()
    src_reg2_value = int(regs_before.get(src_reg2_name), 16)

    if dest_reg_value == src_reg1_value:
        use_var = ses.createUseVar(src_reg1_name, VarType.REG, src_reg1_value)
    else:
        use_var = ses.createUseVar(src_reg2_name, VarType.REG, src_reg2_value)
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    debug(f'{def_var.name} = {use_var.name}')

    ses.exprs.update({def_var: use_var})


def handleCSET(inst_info, ses):
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    ops = inst.getOperands()
    dest_reg = ops[0]

    dest_reg_name = dest_reg.getName()
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)

    use_var = ses.createUseVar(VarName.CONST, VarType.IMM, dest_reg_value)
    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    debug(f'{def_var.name} = {use_var.name}')

    ses.exprs.update({def_var: use_var})


def handleASR(inst_info, ses):
    # 算数右移
    # asr w8, w8, #2
    # asr w8, w8, w9
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    ops = inst.getOperands()
    dest_reg = ops[0]
    src = ops[1:]

    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()

    # 根据源操作数创建变量
    src_vars = []
    for var in src:
        if var.getType() == OPERAND.REG:
            reg_name = var.getName()
            reg_value = int(regs_before.get(reg_name), 16)
            use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
        else:
            use_var = ses.createUseVar(VarName.CONST, VarType.IMM, var.getValue())
        src_vars.append(use_var)

    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    right_expr = Expr('>>', src_vars[0], src_vars[1])

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleLSL(inst_info, ses):
    # 左移, 仅支持如下格式
    # lsl w8, w8, #2
    # lsl w8, w8, w9
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))

    ctx.processing(inst)

    ops = inst.getOperands()
    dest_reg = ops[0]
    src = ops[1:]

    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()

    # 根据源操作数创建变量
    src_vars = []
    for var in src:
        if var.getType() == OPERAND.REG:
            reg_name = var.getName()
            reg_value = int(regs_before.get(reg_name), 16)
            use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
        else:
            use_var = ses.createUseVar(VarName.CONST, VarType.IMM, var.getValue())
        src_vars.append(use_var)

    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    right_expr = Expr('<<', src_vars[0], src_vars[1])

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleAND(inst_info, ses):
    # and W7, W7, #0x7FFFFFFF
    # and w8, w9, w8
    # and X0, X1, X2, LSR #8
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))
        # 需要事先符号化寄存器变量，否则ast解析时获取不到寄存器
        ctx.symbolizeRegister(getattr(ctx.registers, reg))
    # debug(f"symbol register: {ctx.getSymbolicRegisters()}")
    ctx.processing(inst)
    # debug(f"instruction symbolic exprs: {inst.getSymbolicExpressions()}")

    dest_reg = inst.getWrittenRegisters()[0][0]
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()

    # 根据源操作数创建变量
    src_vars = []
    for reg, expr in inst.getReadRegisters():
        # expr 是 ast
        # debug(f"expr: {type(expr)}")
        expr_type = expr.getType()
        if expr_type == AST_NODE.BVSHL:  # 左移
            reg_node, shift_node = expr.getChildren()
            reg_name = str(reg_node).split('_')[0]
            reg_value = int(regs_before.get(reg_name), 16)
            use_var1 = ses.createUseVar(reg_name, VarType.REG, reg_value)
            use_var2 = ses.createUseVar(VarName.CONST, VarType.IMM, int(str(shift_node), 16))
            expr = Expr("<<", use_var1, use_var2)
            src_vars.append(expr)
        elif expr_type in SHIFT_MAP:
            raise Exception("Unsupport shift type in AND instruction")
        else:
            reg_name = reg.getName()
            reg_value = int(regs_before.get(reg_name), 16)
            use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
            src_vars.append(use_var)

    # 也许存在操作数, 例如 add w0, w1, #5
    for imm, expr in inst.getReadImmediates():
        use_var = ses.createUseVar(VarName.CONST, VarType.IMM, imm.getValue())
        src_vars.append(use_var)

    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    right_expr = Expr('&', src_vars[0], src_vars[1])

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleORR(inst_info, ses):
    # orr x8, x8, #0x0000FFFF
    # orr w10, w10, w9
    # orr x0, x1, x2, lsr #8
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))
        # 需要事先符号化寄存器变量，否则ast解析时获取不到寄存器
        ctx.symbolizeRegister(getattr(ctx.registers, reg))
    # debug(f"symbol register: {ctx.getSymbolicRegisters()}")
    ctx.processing(inst)
    # debug(f"instruction symbolic exprs: {inst.getSymbolicExpressions()}")

    dest_reg = inst.getWrittenRegisters()[0][0]
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()

    # 根据源操作数创建变量
    src_vars = []
    for reg, expr in inst.getReadRegisters():
        # expr 是 ast
        # debug(f"expr: {type(expr)}")
        expr_type = expr.getType()
        if expr_type == AST_NODE.BVSHL:  # 左移
            reg_node, shift_node = expr.getChildren()
            reg_name = str(reg_node).split('_')[0]
            reg_value = int(regs_before.get(reg_name), 16)
            use_var1 = ses.createUseVar(reg_name, VarType.REG, reg_value)
            use_var2 = ses.createUseVar(VarName.CONST, VarType.IMM, int(str(shift_node), 16))
            expr = Expr("<<", use_var1, use_var2)
            src_vars.append(expr)
        elif expr_type in SHIFT_MAP:
            raise Exception("Unsupport shift type in ORR instruction")
        else:
            reg_name = reg.getName()
            reg_value = int(regs_before.get(reg_name), 16)
            use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
            src_vars.append(use_var)

    # 也许存在操作数, 例如 add w0, w1, #5
    for imm, expr in inst.getReadImmediates():
        use_var = ses.createUseVar(VarName.CONST, VarType.IMM, imm.getValue())
        src_vars.append(use_var)

    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    right_expr = Expr('|', src_vars[0], src_vars[1])

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleEOR(inst_info, ses):
    # orr x8, x8, #0x0000FFFF
    # orr w10, w10, w9
    # orr x0, x1, x2, lsr #8
    global ctx
    global vm_stack_addr

    regs_before = inst_info.regs_before

    inst = Instruction(int(inst_info.address, 16), bytes.fromhex(inst_info.opcode))
    for reg, value in regs_before.items():
        ctx.setConcreteRegisterValue(getattr(ctx.registers, reg), int(value, 16))
        # 需要事先符号化寄存器变量，否则ast解析时获取不到寄存器
        ctx.symbolizeRegister(getattr(ctx.registers, reg))
    # debug(f"symbol register: {ctx.getSymbolicRegisters()}")
    ctx.processing(inst)
    # debug(f"instruction symbolic exprs: {inst.getSymbolicExpressions()}")

    dest_reg = inst.getWrittenRegisters()[0][0]
    dest_reg_value = ctx.getConcreteRegisterValue(dest_reg)
    dest_reg_name = dest_reg.getName()

    # 根据源操作数创建变量
    src_vars = []
    for reg, expr in inst.getReadRegisters():
        # expr 是 ast
        # debug(f"expr: {type(expr)}")
        expr_type = expr.getType()
        if expr_type == AST_NODE.BVSHL:  # 左移
            reg_node, shift_node = expr.getChildren()
            reg_name = str(reg_node).split('_')[0]
            reg_value = int(regs_before.get(reg_name), 16)
            use_var1 = ses.createUseVar(reg_name, VarType.REG, reg_value)
            use_var2 = ses.createUseVar(VarName.CONST, VarType.IMM, int(str(shift_node), 16))
            expr = Expr("<<", use_var1, use_var2)
            src_vars.append(expr)
        elif expr_type in SHIFT_MAP:
            raise Exception("Unsupport shift type in EOR instruction")
        else:
            reg_name = reg.getName()
            reg_value = int(regs_before.get(reg_name), 16)
            use_var = ses.createUseVar(reg_name, VarType.REG, reg_value)
            src_vars.append(use_var)

    # 也许存在操作数, 例如 add w0, w1, #5
    for imm, expr in inst.getReadImmediates():
        use_var = ses.createUseVar(VarName.CONST, VarType.IMM, imm.getValue())
        src_vars.append(use_var)

    def_var = ses.createDefVar(dest_reg_name, VarType.REG, dest_reg_value)

    right_expr = Expr('^', src_vars[0], src_vars[1])

    debug(f'{def_var.name} = {right_expr}')

    ses.exprs.update({def_var: right_expr})


def handleInst(inst_info: InstructionInfo, ses: SliceExprState):
    mnemonic = inst_info.inst_str.split(" ")[0]

    if mnemonic in ['ldrb', 'ldr', 'ldrsh', 'ldrh', 'ldrsw']:  # 单寄存器load
        return handleLDR(inst_info, ses)
    elif mnemonic in ['strb', 'str', 'stur']:  # 单寄存器store
        handleSTR(inst_info, ses)
    elif mnemonic in ['ldp']:  # 双寄存器load
        handleLDP(inst_info, ses)
    elif mnemonic in ['stp']:
        handleSTP(inst_info, ses)
    elif mnemonic in ['add']:  # 加法
        handleADD(inst_info, ses)
    elif mnemonic in ['sub']:
        handleSUB(inst_info, ses)
    elif mnemonic in ['mul']:
        handleMUL(inst_info, ses)
    elif mnemonic in ['udiv', 'sdiv', 'div']:
        handleDIV(inst_info, ses)
    elif mnemonic in ['msub']:
        handleMSUB(inst_info, ses)
    elif mnemonic in ['and']:
        handleAND(inst_info, ses)
    elif mnemonic in ['orr']:
        handleORR(inst_info, ses)
    elif mnemonic in ['eor']:
        handleEOR(inst_info, ses)
    elif mnemonic in ['asr']:
        handleASR(inst_info, ses)
    elif mnemonic in ['lsl']:
        handleLSL(inst_info, ses)
    elif mnemonic in ['mov', 'movz', 'movn']:
        handleMOV(inst_info, ses)
    elif mnemonic in ['movk']:
        handleMOVK(inst_info, ses)
    elif mnemonic in ['adrp']:
        handleADRP(inst_info, ses)
    elif mnemonic in ['b', 'bl', 'br', 'blr', 'b.ge', 'b.ne', 'b.eq', 'b.lt']:
        pass
    elif mnemonic in ['sxth']:
        handleSXTH(inst_info, ses)
    elif mnemonic in ['cmp']:
        pass
    elif mnemonic in ['cbnz']:  # 比较-跳转指令
        pass
    elif mnemonic in ['csel']:
        handleCSEL(inst_info, ses)
    elif mnemonic in ['cset']:
        handleCSET(inst_info, ses)
    
    else:
        raise Exception(f"Unsupport Instruction: {inst_info.inst_str}. You should add handle method in handleInst() function!")


def liftToExpr(instInfo_list: List[InstructionInfo], ses: SliceExprState, vm_addr: int):
    global ctx
    global vm_stack_addr

    ctx.setAstRepresentationMode(AST_REPRESENTATION.PCODE)
    vm_stack_addr = vm_addr

    for inst_info in instInfo_list:
        debug(inst_info.toString())
        handleInst(inst_info, ses)
