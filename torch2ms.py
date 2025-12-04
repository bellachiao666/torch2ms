import difflib
import json
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Optional

import libcst as cst
from libcst import helpers as cst_helpers
from libcst import metadata


with open("api_map.json", "r", encoding="utf8") as f:
    API_MAP = json.load(f)["apis"]


def _literal_to_expr(value):
    """
    将 API 默认值转成 LibCST 表达式。
    """
    return cst.parse_expression(repr(value))


class _MindSporePrefixDetector(cst.CSTVisitor):
    """
    读取导入语句，推断 MindSpore 前缀（兼容 mint）。
    """

    def __init__(self) -> None:
        self.prefix = None

    def visit_Import(self, node: cst.Import) -> None:
        if self.prefix:
            return
        for name in node.names:
            module = cst_helpers.get_full_name_for_node(name.name)
            alias = getattr(name, "evaluated_name", None) or module
            if module == "mindspore.mint":
                self.prefix = f"{alias}.nn"
                return
            if module == "mindspore.mint.nn":
                self.prefix = alias
                return
            if module == "mindspore":
                self.prefix = f"{alias}.nn"
                return
            if module == "mindspore.nn":
                self.prefix = alias
                return

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if self.prefix or node.module is None:
            return
        module_name = cst_helpers.get_full_name_for_node(node.module)
        if module_name not in ("mindspore", "mindspore.mint"):
            return
        for name in node.names:
            if isinstance(name, cst.ImportStar):
                continue
            alias = getattr(name, "evaluated_name", None) or name.name.value
            if alias == "nn":
                self.prefix = "nn"
                return


def detect_mindspore_prefix(module: cst.Module) -> str:
    """
    根据导入语句寻找 MindSpore 前缀，默认为 nn。
    """
    detector = _MindSporePrefixDetector()
    module.visit(detector)
    return detector.prefix or "nn"


class TorchToMindSporeTransformer(cst.CSTTransformer):
    """
    使用 LibCST 将 PyTorch API 调用改写为 MindSpore 调用。
    """

    METADATA_DEPENDENCIES = (metadata.ParentNodeProvider,)

    def __init__(self, api_map, ms_prefix: str) -> None:
        super().__init__()
        self.api_map = api_map
        self.ms_prefix = ms_prefix
        self.notes_by_stmt = defaultdict(list)
        self.api_by_class = {
            conf["pytorch"].split(".")[-1]: conf for conf in api_map.values()
        }

    def _find_enclosing_stmt(self, node: cst.CSTNode) -> Optional[cst.CSTNode]:
        parent = self.get_metadata(metadata.ParentNodeProvider, node, None)
        while parent is not None and not isinstance(parent, cst.SimpleStatementLine):
            parent = self.get_metadata(metadata.ParentNodeProvider, parent, None)
        return parent

    def _record_note(self, node: cst.CSTNode, notes) -> None:
        stmt = self._find_enclosing_stmt(node)
        if stmt is not None:
            self.notes_by_stmt[stmt].extend(notes)

    def _reconstruct_args(self, call: cst.Call, api_conf):
        params = api_conf["params"]
        if not params:
            return call.args, []

        positional_values = []
        keyword_values = {}
        for arg in call.args:
            if arg.star:
                return call.args, []  # 避免破坏 *args/**kwargs
            if arg.keyword is None:
                positional_values.append(arg.value)
            else:
                keyword_values[arg.keyword.value] = arg.value

        ms_args = OrderedDict()
        mismatch_notes = []

        for i, val in enumerate(positional_values):
            if i >= len(params):
                break
            ms_name = params[i]["mindspore"]["name"]
            if ms_name:
                ms_args[ms_name] = val

        for p in params:
            pt = p["pytorch"]
            ms = p["mindspore"]
            pt_name = pt.get("name")
            ms_name = ms.get("name")
            pt_def = pt.get("default")
            ms_def = ms.get("default")

            if ms_name is None:
                mismatch_notes.append(f"没有对应的mindspore参数 '{pt_name}'")
                continue

            if pt_name and pt_name in keyword_values:
                ms_args[ms_name] = keyword_values[pt_name]
                if pt_name != ms_name:
                    mismatch_notes.append(
                        f"默认参数名不一致: {ms_name} (PyTorch={pt_name}, MindSpore={ms_name})"
                    )
                continue

            if (
                pt_name is not None
                and pt_def is not None
                and ms_def is not None
                and pt_def != ms_def
            ):
                ms_args[ms_name] = _literal_to_expr(pt_def)
                mismatch_notes.append(
                    f"默认值不一致: {ms_name} (PyTorch={pt_def}, MindSpore={ms_def})"
                )

        new_args = [
            cst.Arg(keyword=cst.Name(k), value=v)
            for k, v in ms_args.items()
        ]
        return new_args, mismatch_notes

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        full_name = cst_helpers.get_full_name_for_node(updated_node.func)
        if not full_name:
            return updated_node

        api_conf = self.api_by_class.get(full_name.split(".")[-1])
        if not api_conf:
            return updated_node

        new_args, notes = self._reconstruct_args(updated_node, api_conf)
        ms_class = api_conf["mindspore"].split(".")[-1]
        new_callee = cst.parse_expression(f"{self.ms_prefix}.{ms_class}")
        new_call = updated_node.with_changes(func=new_callee, args=new_args)

        if notes:
            self._record_note(original_node, notes)

        return new_call

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.SimpleStatementLine:
        notes = self.notes_by_stmt.get(original_node)
        if not notes:
            return updated_node

        comment_text = "; ".join(dict.fromkeys(notes))
        tw = updated_node.trailing_whitespace
        if tw.comment:
            existing = tw.comment.value.lstrip("#").strip()
            comment_text = f"{existing}; {comment_text}"

        new_trailing = cst.TrailingWhitespace(
            whitespace=cst.SimpleWhitespace("  "),
            comment=cst.Comment(f"# {comment_text}"),
            newline=tw.newline,
        )
        return updated_node.with_changes(trailing_whitespace=new_trailing)


def convert_code(code: str) -> str:
    """
    将整段 PyTorch 源码转换为 MindSpore 源码（LibCST 版本）。
    """
    module = cst.parse_module(code)
    prefix = detect_mindspore_prefix(module)
    wrapper = metadata.MetadataWrapper(module)
    new_module = wrapper.visit(TorchToMindSporeTransformer(API_MAP, prefix))
    return new_module.code


def generate_diff(old: str, new: str) -> str:
    """
    生成原文件和新文件之间的 diff。
    """
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

    base, ext = os.path.splitext(filename)
    new_filename = f"{base}_ms{ext}"

    with open(new_filename, "w", encoding="utf8") as f:
        f.write(result)

    diff = generate_diff(code, result)
    print("=== 转换 DIFF 开始 ===")
    print(diff)
    print("=== 转换 DIFF 结束 ===")

    diff_filename = f"diff_({os.path.basename(filename)}-{os.path.basename(new_filename)}).diff"
    diff_path = os.path.join(os.path.dirname(filename) or ".", diff_filename)
    with open(diff_path, "w", encoding="utf8") as f:
        f.write(diff)
    print(f"已保存 diff 到: {diff_path}")

    print(f"\n已生成新文件: {new_filename}")
