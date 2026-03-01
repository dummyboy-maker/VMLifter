
"""
栈式虚拟机入口特征：
1、栈提升： sub sp, #xxx
2、寄存器环境保存：  连续的stp或者str
"""
from typing import List

from unidbgTraceParser import InstructionInfo


def extractVMCode(instInfo_list: List[InstructionInfo]):
    """
    删除栈式虚拟机入口特征和出口特征，仅保留正文
    :param instInfo_list:
    :return:
    """
    # 假定头尾对称式 sub stp stp ...       ... ldp ldp add 额外一个 ret
    first_inst = instInfo_list[0]
    if not first_inst.inst_str.startswith("sub sp"):
        return instInfo_list

    count = 1
    while 1:
        inst = instInfo_list[count]
        if inst.inst_str.startswith("stp"):
            count += 1
        else:
            break

    start = count
    end = len(instInfo_list) - (count + 1)

    instInfo_list = instInfo_list[start: end]

    # print(instInfo_list[0].inst_str)
    # print(instInfo_list[-1].inst_str)

    return instInfo_list

