import time

from codeSlicing import loadStoreSlicing
from opWrapper import SliceExprState
from sliceTransform import liftToExpr
from unidbgTraceParser import parse_lines
from vmExtract import extractVMCode

if __name__ == '__main__':
    # vm_addr = 0x127090c0  # for sub_1cc3c0

    # vm_addr = 0x126ee000    # for sub_1de514
    # file_name = "sub_1de514.txt"
    # out_file_name = "sub_1de514_expr.txt"

    vm_addr = 0x12716280  # for sub_1d52d4
    file_name = "sub_1d52d4.txt"
    out_file_name = "sub_1d52d4_expr.txt"

    start = time.time()
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.readlines()

    instInfo_list = parse_lines(data)

    # 最后一般是将参数放入x0,x1,x2..., 然后恢复上下文环境, 因此这部分也可以不还原
    # 那么不启用extractVMCode的话, 切片就不会记录参数放入x0,x1,x2的过程
    # instInfo_list = extractVMCode(instInfo_list)

    # 进行 load - store 代码切片
    slice_list = loadStoreSlicing(instInfo_list, vm_addr)

    # 创建切片表达式状态 实例
    ses = SliceExprState()

    expr_file = open(out_file_name, "w")
    # 遍历切片，将其还原出表达式
    for i in range(len(slice_list)):
        ls_slice = slice_list[i]
        print(f"-------------第 {i} 个切片-------------")
        if len(ls_slice) == 0:
            continue
        for ins in ls_slice:
            print(ins.toString())
        # 指令提升
        liftToExpr(ls_slice, ses, vm_addr)

        # 表达式回溯
        last_key = list(ses.exprs.keys())[-1]
        last_expr = ses.exprs[last_key]
        print(f'对 {last_key} = {last_key} 进行表达式回溯')
        new_expr = ses.backTraceExpr(last_expr)

        # 具体化表达式的结果
        concreteExpr = ses.concretizeExprValue(new_expr)
        print(f"回溯后的表达式为: {last_key} = {concreteExpr}")
        final_result = hex(eval(concreteExpr.replace('/', '//')) & ((1 << 64) - 1))

        resultExpr = f'{last_key} = {new_expr} = {concreteExpr} = {final_result}\t{hex(last_key.value) == final_result}'
        # print(f"最终的简化表达式为: {resultExpr}")
        expr_file.write(resultExpr + "\n")

    end = time.time()

    print(f"总计花费 {end - start}s")
    print(f"总计 {len(slice_list)} 个 load-store 切片")
