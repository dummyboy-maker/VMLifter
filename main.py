import argparse
import time
from pathlib import Path

from codeSlicing import loadStoreSlicing
from opWrapper import SliceExprState, isContainMem
from sliceTransform import liftToExpr
from toolsFunc import debug
from unidbgTraceParser import parse_lines
from vmExtract import extractVMCode

import toolsFunc


def auto_int(x):
    """自动识别进制转换函数"""
    try:
        return int(x, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"无效的数字输入: '{x}'。请使用十进制或带前缀的十六进制(0x)。")


def main():
    # 1. 初始化解析器
    parser = argparse.ArgumentParser(
        description="VMLifter: 虚拟指令追踪与提升工具"
    )

    # 2. 添加 --vm-addr 参数
    # 使用 lambda 处理十六进制字符串，使其直接转换为 int
    parser.add_argument(
        "-v",
        "--vm-addr",
        type=auto_int,
        required=True,
        help="虚拟栈的基地址 (例如: 0x127000)"
    )

    # 3. 添加 --file 参数
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="unidbg trace文件路径"
    )

    # 4. 添加 --debug 可选参数
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",  # 只要出现该参数，值就为 True
        help="显示详细的切片语义提升过程"
    )

    # 5. 解析参数
    args = parser.parse_args()
    file_path = Path(args.file)
    vm_addr = args.vm_addr
    if args.debug:
        toolsFunc.DEBUG = True

    pure_name = file_path.stem  # 获取不带后缀的主文件名 (方便生成输出文件)
    parent_dir = file_path.parent   # 获取父目录路径
    out_file_name = parent_dir / f"{pure_name}_expr.txt"

    print(f"[*] 目标文件: {file_path}")
    print(f"[*] 虚拟栈基址: {hex(vm_addr)}")

    # 6. 开始解析trace文件
    start(vm_addr, file_path, out_file_name)


def start(vm_addr, file_path, out_file_name):

    start_time = time.time()
    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    print("[*] 解析指令轨迹中...")
    instInfo_list = parse_lines(data)

    # 最后一般是将参数放入x0,x1,x2..., 然后恢复上下文环境, 因此这部分也可以不还原
    # 那么不启用extractVMCode的话, 切片就不会记录参数放入x0,x1,x2的过程
    # instInfo_list = extractVMCode(instInfo_list)

    # 进行 load - store 代码切片
    print("[*] 进行load-store切片...")
    slice_list = loadStoreSlicing(instInfo_list, vm_addr)
    slice_len = len(slice_list)
    print(f"[*] 总计 {slice_len} 个load-store切片")

    # 创建切片表达式状态 实例
    ses = SliceExprState()

    expr_file = open(out_file_name, "w")
    # 遍历切片，将其还原出表达式
    print("[*] 进行load-store切片语义提升...")
    for i in range(slice_len):
        print(f"\r[*] 正在处理: 第 {i+1}/{slice_len} 个 load-store 切片...", end="", flush=True)
        ls_slice = slice_list[i]
        if len(ls_slice) == 0:
            continue

        for ins in ls_slice:
            debug(ins.toString())

        # 指令提升
        liftToExpr(ls_slice, ses, vm_addr)

        # 表达式回溯
        last_key = list(ses.exprs.keys())[-1]
        last_expr = ses.exprs[last_key]
        debug(f'对 {last_key} = {last_expr} 进行表达式回溯')
        new_expr = ses.backTraceExpr(last_expr)
        debug(f'回溯后的表达式为: {last_key} = {new_expr}')

        # 具体化表达式的结果
        concreteExpr = ses.concretizeExprValue(new_expr)
        debug(f"表达式带入具体值: {last_key} = {concreteExpr}")

        # 简化表达式输出
        resultExpr = f'{last_key}'
        if isContainMem(new_expr):# 存在内存数，需要展示; 如果全是立即数，则不予展示
            resultExpr += f'= {new_expr}'
        if " " in concreteExpr:   # 存在计算过程
            resultExpr += f'= {concreteExpr}'
        resultExpr += f'= {hex(last_key.value)}'

        debug(f"最终的简化表达式为: {resultExpr}")

        expr_file.write(resultExpr + "\n")

    end_time = time.time()
    print()
    print(f"[*] 切片语义提升完毕！生成的表达式文件位于: {out_file_name}")
    print(f"[*] 总计花费 {end_time - start_time}s")


if __name__ == '__main__':

    main()
