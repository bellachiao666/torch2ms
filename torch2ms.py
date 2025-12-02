import json
import re
import sys
import difflib
import os


with open("api_map.json", "r", encoding="utf8") as f:
    API_MAP = json.load(f)["apis"]


def parse_args(arg_str):
    '''
        capture 参数字符串，分离位置参数和关键字参数
        例如：
        "3, 64, 3, bias=False" 解析为
        {
            "__positional__": ["3","64","3"],
            "bias": "False"
        }
    '''
    args = {"__positional__": []}

    if not arg_str.strip():
        return args

    parts = [p.strip() for p in arg_str.split(",") if p.strip()]

    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            args[k.strip()] = v.strip()
        else:
            args["__positional__"].append(part)

    return args


def reconstruct_args(api_conf, pytorch_args):
    '''
        根据 api_conf 和 pytorch_args 重构 MindSpore 参数字符串
        同时记录默认值不一致的参数
    '''
    ms_args = {}
    mismatch_notes = []

    positional = pytorch_args.get("__positional__", [])
    params = api_conf["params"]

    # 位置参数映射
    for i, val in enumerate(positional):
        if i >= len(params):
            break
        p = params[i]
        ms_name = p["mindspore"]["name"]
        if ms_name:
            ms_args[ms_name] = val

    # 合并关键字参数处理 + 默认值 mismatch
    for p in params:
        pt = p["pytorch"]
        ms = p["mindspore"]
        pt_name = pt["name"]
        ms_name = ms["name"]

        # ------------------------------------------
        # ① MindSpore 无对应参数 → 记录并跳过
        # ------------------------------------------
        if ms_name is None:
            mismatch_notes.append(f"没有对应的mindspore参数 '{pt_name}'")
            continue

        # ------------------------------------------
        # ② 若 PyTorch 显式写了参数 → 优先直接映射
        # ------------------------------------------
        if pt_name and pt_name in pytorch_args:
            ms_args[ms_name] = pytorch_args[pt_name]
            if pt_name != ms_name:
                mismatch_notes.append(
                    f"默认参数名不一致: {ms_name} (PyTorch={pt_name}, MindSpore={ms_name})"
                )
            # continue

        # ------------------------------------------
        # ③ PyTorch 未写，但默认值不同 → 自动补齐 PyTorch 默认值
        # ------------------------------------------
        if (
            pt_name is not None
            and pt["default"] != ms["default"]
            # and ms_name not in ms_args
        ):
            v = pt["default"]
            if isinstance(v, str):
                v = f'"{v}"'

            ms_args[ms_name] = v
            mismatch_notes.append(
                f"默认值不一致: {ms_name} (PyTorch={pt['default']}, MindSpore={ms['default']})"
            )

        # ------------------------------------------
        # ④ 其他情况：默认值相同、且用户没写 → 什么也不做
        # ------------------------------------------
        
    # print(ms_args)
    arg_str = ", ".join(f"{k}={v}" for k, v in ms_args.items())

    note_str = ""
    if mismatch_notes:
        note_str = " \n # " + "; \n # ".join(mismatch_notes)

    return arg_str, note_str



def detect_mindspore_prefix(code):
    '''
        检测 mindspore 的导入前缀
    '''
    m = re.search(r"import\s+mindspore\s+as\s+(\w+)", code)
    if m:
        return f"{m.group(1)}.nn"

    m = re.search(r"import\s+mindspore\.nn\s+as\s+(\w+)", code)
    if m:
        return m.group(1)

    if re.search(r"from\s+mindspore\s+import\s+nn", code):
        return "nn"

    return "nn"  # 默认


def convert_call(code_line, prefix):
    '''
        转换单行代码
    '''
    for api_name, api_conf in API_MAP.items():
        pt_class = api_conf["pytorch"].split(".")[-1]
        ms_class = api_conf["mindspore"].split(".")[-1]

        # 匹配模式：任意前缀.类名(...)
        pattern = rf"(\w+\.)?{pt_class}\((.*)\)"
        m = re.search(pattern, code_line)
        if not m:
            continue

        # full_prefix = m.group(1)  # 比如 "nn." 
        arg_str = m.group(2)

        pytorch_args = parse_args(arg_str)

        ms_arg_str, note = reconstruct_args(api_conf, pytorch_args)
        new_line = f"{prefix}.{ms_class}({ms_arg_str}){note}"

        # 整段替换，而不是只替换类名
        return re.sub(pattern, new_line, code_line)

    return code_line



def convert_code(code):
    '''
        转换整段代码'''
    prefix = detect_mindspore_prefix(code)

    output = []
    for line in code.split("\n"):
        new_line = convert_call(line, prefix)
        output.append(new_line)
    return "\n".join(output)


def generate_diff(old, new):
    '''
        生成原文件和新文件之间的 diff
    '''
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile="pytorch",
        tofile="mindspore",
        lineterm=""
    )
    return "".join(diff)



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python torch2ms.py input.py")
        sys.exit(0)

    filename = sys.argv[1]

    with open(filename, "r", encoding="utf8") as f:
        code = f.read()

    result = convert_code(code)

    # 生成新文件名
    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_ms{ext}"

    # 写入新文件
    with open(new_filename, "w", encoding="utf8") as f:
        f.write(result)

    # 打印 diff
    diff = generate_diff(code, result)
    print("=== 转换 DIFF 开始 ===")
    print(diff)
    print("=== 转换 DIFF 结束 ===")
    print(f"\n已生成新文件: {new_filename}")