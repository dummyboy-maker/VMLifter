from copy import copy
from typing import List
from triton import *

from toolsFunc import debug
from unidbgTraceParser import InstructionInfo


def getMemoryBaseRegName(inst: Instruction):
    # Triton 会自动识别所有的操作数
    for op in inst.getOperands():
        if op.getType() == OPERAND.MEM:
            # 直接获取内存操作数的基址寄存器对象
            base_reg = op.getBaseRegister()
            return base_reg.getName()
    return None


def loadStoreSlicing(insn_info_list: List[InstructionInfo], vm_stack_addr: int) -> List[List[InstructionInfo]]:
    """
                    真实环境中的内存数据在虚拟机中的生命周期
                       load from real environment        (start)
                                  |
                                  |
                       |——>|——>register——>|——>|
                       |   |             |    |
                       |   |——operation——|    |
                       |                      |
                       <———————vmstack<————————
                                  |
                                  |
                               regiter
                                  |
                                  |
                       store into real environment        (end)
    """
    slice_list = []  # List[List[InstructionInfo]]
    inst_num = len(insn_info_list)

    index = 0
    flag = False
    ls_slice = []  # List[InstructionInfo]
    ctx = TritonContext(ARCH.AARCH64)

    while index < inst_num:
        inst_info = insn_info_list[index]
        mnemonic = inst_info.inst_str.split(" ")[0]
        if flag:
            ls_slice.append(insn_info_list[index])
        else:
            if mnemonic.startswith("ld"):  # load instruciton
                inst = Instruction(bytes.fromhex(inst_info.opcode))
                ctx.processing(inst)
                base_name = getMemoryBaseRegName(inst)
                base_value = inst_info.regs_before.get(base_name)
                if int(base_value, 16) != vm_stack_addr:
                    debug(f"find load real memory: {inst.getDisassembly()}")
                    # start new record
                    flag = True
                    ls_slice.append(insn_info_list[index])

        if mnemonic.startswith("st"):  # store instruciton
            inst = Instruction(bytes.fromhex(inst_info.opcode))
            ctx.processing(inst)
            base_name = getMemoryBaseRegName(inst)
            base_value = inst_info.regs_before.get(base_name)
            if int(base_value, 16) != vm_stack_addr:
                # end record
                debug(len(ls_slice))
                slice_list.append(copy(ls_slice))
                ls_slice = []
                flag = False

        index += 1

    # 未找到store指令仍加入，前提是使用extractVMCode
    # slice_list.append(copy(ls_slice))

    return slice_list
