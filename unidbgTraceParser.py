import re
import time
from typing import List

log_pattern = re.compile(
    r'\[[^\]]*\]'  # 跳过时间戳
    r'\[(?P<module>[^ ]+)\s'  # 匹配模块名，遇到空格停止
    r'(?P<offset>0x[0-9a-f]+)\]\s'  # 匹配偏移量
    r'\[(?P<opcode>[0-9a-f]+)\]\s'  # 匹配机器码
    r'(?P<addr>0x[0-9a-f]+):\s'  # 匹配PC
    r'"(?P<inst_str>.*?)"'  # 匹配指令文本
    r'(?P<change_state>.*)'  # 匹配寄存器状态
)

reg_pattern = re.compile(r'([a-zA-Z0-9]+)=(0x[0-9a-f]+|[0-9a-f]+)')


class InstructionInfo:

    def __init__(self, opcode: str, address: str, inst_str: str, regs_before: dict=None, regs_after: dict=None):
        self.opcode = opcode
        self.address = address
        self.inst_str = inst_str
        self.regs_before = regs_before
        self.regs_after = regs_after

    def toString(self):
        regs_Info = ""
        for reg, value in self.regs_before.items():
            regs_Info += "%s=%s " % (reg, value)
        regs_Info += "=> "
        for reg, value in self.regs_after.items():
            regs_Info += "%s=%s " % (reg, value)

        return '%s: "%s" %s' % (self.address, self.inst_str, regs_Info)


def parse_line(line: str) -> InstructionInfo:
    match = log_pattern.search(line)
    if not match:
        return None

    data = match.groupdict()

    change_state = data['change_state']
    if "=>" in change_state:
        left, right = change_state.split("=>")
    else:
        left, right = change_state, ""

    regs_before = {k.lower(): v for k, v in reg_pattern.findall(left)}
    regs_after = {k.lower(): v for k, v in reg_pattern.findall(right)}

    inst_info = InstructionInfo(data['opcode'],
                                data['addr'], data['inst_str'],
                                regs_before=regs_before,
                                regs_after=regs_after)

    return inst_info


def parse_lines(lines: List[str]) -> List[InstructionInfo]:
    inst_info_list = []
    for line in lines:
        match = log_pattern.search(line)
        if not match:
            continue
        data = match.groupdict()

        change_state = data['change_state']
        if "=>" in change_state:
            left, right = change_state.split("=>")
        else:
            left, right = change_state, ""

        regs_before = {k.lower(): v for k, v in reg_pattern.findall(left)}
        regs_after = {k.lower(): v for k, v in reg_pattern.findall(right)}

        inst_info = InstructionInfo(data['opcode'],
                                    data['addr'], data['inst_str'],
                                    regs_before=regs_before,
                                    regs_after=regs_after)

        inst_info_list.append(inst_info)

    return inst_info_list


if __name__ == '__main__':

    start = time.time()
    with open('taintAnalyzeResult.txt', 'r', encoding='utf-8') as f:
        data = f.readlines()

    instInfo_list = parse_lines(data)
    end = time.time()

    print(f"总计花费 {end - start}s")
    for info in instInfo_list:
        print(info.toString())
