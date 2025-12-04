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


class _MindSporeImportCollector(cst.CSTVisitor):
    """
    读取导入语句，记录所有 MindSpore 模块前缀（不限于 nn）。
    """

    def __init__(self) -> None:
        self.alias_by_module = {}

    def _maybe_record(self, module: str, alias: str) -> None:
        if not module:
            return
        if not module.startswith("mindspore"):
            return
        self.alias_by_module[module] = alias

    def visit_Import(self, node: cst.Import) -> None:
        for name in node.names:
            module = cst_helpers.get_full_name_for_node(name.name)
            alias = (
                name.asname.name.value
                if getattr(name, "asname", None)
                else getattr(name, "evaluated_name", None)
            ) or module
            self._maybe_record(module, alias)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return
        module_name = cst_helpers.get_full_name_for_node(node.module)
        if not module_name or not module_name.startswith("mindspore"):
            return

        for name in node.names:
            if isinstance(name, cst.ImportStar):
                continue
            alias = (
                name.asname.name.value
                if getattr(name, "asname", None)
                else getattr(name, "evaluated_name", None)
            ) or name.name.value
            full_module = f"{module_name}.{name.name.value}"
            self._maybe_record(full_module, alias)


def collect_mindspore_aliases(module: cst.Module) -> dict:
    """
    根据导入语句收集 MindSpore 模块前缀映射。
    """
    collector = _MindSporeImportCollector()
    module.visit(collector)
    return collector.alias_by_module


class _TorchImportCollector(cst.CSTVisitor):
    """
    收集 torch 模块别名，记录 alias -> module 完整路径。
    """

    def __init__(self) -> None:
        self.module_by_alias = {}

    def _maybe_record(self, module: str, alias: str) -> None:
        if not module or not module.startswith("torch"):
            return
        self.module_by_alias[alias] = module

    def visit_Import(self, node: cst.Import) -> None:
        for name in node.names:
            module = cst_helpers.get_full_name_for_node(name.name)
            alias = (
                name.asname.name.value
                if getattr(name, "asname", None)
                else getattr(name, "evaluated_name", None)
            ) or module
            self._maybe_record(module, alias)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module is None:
            return
        module_name = cst_helpers.get_full_name_for_node(node.module)
        if not module_name or not module_name.startswith("torch"):
            return

        for name in node.names:
            if isinstance(name, cst.ImportStar):
                continue
            alias = (
                name.asname.name.value
                if getattr(name, "asname", None)
                else getattr(name, "evaluated_name", None)
            ) or name.name.value
            full_module = f"{module_name}.{name.name.value}"
            self._maybe_record(full_module, alias)


def collect_torch_aliases(module: cst.Module) -> dict:
    collector = _TorchImportCollector()
    module.visit(collector)
    return collector.module_by_alias


class _TorchAliasAssignmentCollector(cst.CSTVisitor):
    """
    记录简单赋值形成的别名映射，例如: `myconv = nn.Conv2d`。
    """

    def __init__(self, api_by_class: dict, pt_aliases: dict) -> None:
        self.api_by_class = api_by_class
        self.pt_aliases = pt_aliases
        self.alias_to_pt = {}
        self.alias_to_pt_wrapped = {}


    def _resolve_pt_full_name(self, full_name: Optional[str]) -> Optional[str]:
        if not full_name:
            return None

        parts = full_name.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            module = self.pt_aliases.get(prefix)
            if module:
                suffix = parts[i:]
                if suffix:
                    return ".".join([module, *suffix])
                return module

        if full_name.startswith("torch"):
            return full_name
        return None

    def _maybe_record_alias(self, target: cst.CSTNode, value: cst.CSTNode) -> None:
        if not isinstance(target, cst.Name):
            return
        if value is None:
            return
        if isinstance(value, cst.Call):
            return

        alias_name = target.value
        value_full = cst_helpers.get_full_name_for_node(value)
        root = value_full.split(".")[0] if value_full else None
        if root in {"self", "cls"}:
            return

        pt_full_name = self._resolve_pt_full_name(value_full)
        if pt_full_name and pt_full_name.startswith("torch"):
            self.alias_to_pt[alias_name] = pt_full_name
            return

        if value_full:
            class_name = value_full.split(".")[-1]
            api_conf = self.api_by_class.get(class_name)
            if api_conf:
                self.alias_to_pt_wrapped[alias_name] = api_conf["pytorch"]

    def visit_Assign(self, node: cst.Assign) -> None:
        if len(node.targets) != 1:
            return
        self._maybe_record_alias(node.targets[0].target, node.value)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        self._maybe_record_alias(node.target, node.value)


def collect_torch_assignment_aliases(module: cst.Module, api_by_class: dict, pt_aliases: dict) -> dict:
    collector = _TorchAliasAssignmentCollector(api_by_class, pt_aliases)
    module.visit(collector)
    return collector.alias_to_pt, collector.alias_to_pt_wrapped


class TorchToMindSporeTransformer(cst.CSTTransformer):
    """
    使用 LibCST 将 PyTorch API 调用改写为 MindSpore 调用。
    """

    METADATA_DEPENDENCIES = (metadata.ParentNodeProvider,)

    def __init__(
        self,
        api_map,
        ms_aliases: Optional[dict] = None,
        pt_aliases: Optional[dict] = None,
        pt_assignment_aliases: Optional[dict] = None,
        pt_assignment_wrapped: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.api_map = api_map
        self.ms_aliases = ms_aliases or {}
        self.pt_aliases = pt_aliases or {}
        self.pt_assignment_aliases = pt_assignment_aliases or {}
        self.pt_assignment_wrapped = pt_assignment_wrapped or {}
        self.notes_by_stmt = defaultdict(list)
        self.api_by_class = {
            conf["pytorch"].split(".")[-1]: conf for conf in api_map.values()
        }
        self.api_by_pt_full = {
            conf["pytorch"]: conf for conf in api_map.values()
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

    def _resolve_ms_full_name(self, full_path: str) -> str:
        """
        使用导入的别名还原 MindSpore API 全路径。
        """
        parts = full_path.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            alias = self.ms_aliases.get(prefix)
            if alias:
                suffix = parts[i:]
                if suffix:
                    return ".".join([alias, *suffix])
                return alias
        return full_path

    def _resolve_pt_full_name(self, full_path: Optional[str]) -> Optional[str]:
        """
        将调用名还原成完整的 PyTorch 路径，避免将同名自定义方法误判为 torch API。
        """
        if not full_path:
            return None

        if full_path in self.pt_assignment_aliases:
            return self.pt_assignment_aliases[full_path]

        parts = full_path.split(".")
        for i in range(len(parts), 0, -1):
            prefix = ".".join(parts[:i])
            module = self.pt_aliases.get(prefix)
            if module:
                suffix = parts[i:]
                if suffix:
                    return ".".join([module, *suffix])
                return module

        if full_path.startswith("torch"):
            return full_path
        return None

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
                mismatch_notes.append(f"'{api_conf['pytorch']}':没有对应的mindspore参数 '{pt_name}';")
                continue

            if pt_name and pt_name in keyword_values:
                ms_args[ms_name] = keyword_values[pt_name]
                if pt_name != ms_name:
                    mismatch_notes.append(
                        f"'{api_conf['pytorch']}':默认参数名不一致: {ms_name} (PyTorch={pt_name}, MindSpore={ms_name});"
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
        pt_full_name = self._resolve_pt_full_name(full_name)
        api_conf = None

        if pt_full_name:
            api_conf = self.api_by_pt_full.get(pt_full_name) or self.api_by_class.get(pt_full_name.split(".")[-1])
            if api_conf:
                new_args, notes = self._reconstruct_args(updated_node, api_conf)
                ms_full_name = self._resolve_ms_full_name(api_conf["mindspore"])
                new_callee = cst.parse_expression(ms_full_name)
                new_call = updated_node.with_changes(func=new_callee, args=new_args)
                if notes:
                    self._record_note(original_node, notes)
                return new_call

        wrapper_pt_full = self.pt_assignment_wrapped.get(full_name)
        if wrapper_pt_full:
            api_conf = self.api_by_pt_full.get(wrapper_pt_full) or self.api_by_class.get(wrapper_pt_full.split(".")[-1])
            if api_conf:
                new_args, notes = self._reconstruct_args(updated_node, api_conf)
                new_call = updated_node.with_changes(args=new_args)
                if notes:
                    self._record_note(original_node, notes)
                return new_call

        return updated_node

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
    ms_aliases = collect_mindspore_aliases(module)
    api_by_class = {
        conf["pytorch"].split(".")[-1]: conf for conf in API_MAP.values()
    }
    pt_aliases = collect_torch_aliases(module)
    pt_assignment_aliases, pt_assignment_wrapped = collect_torch_assignment_aliases(module, api_by_class, pt_aliases)
    wrapper = metadata.MetadataWrapper(module)
    new_module = wrapper.visit(
        TorchToMindSporeTransformer(
            API_MAP,
            ms_aliases=ms_aliases,
            pt_aliases=pt_aliases,
            pt_assignment_aliases=pt_assignment_aliases,
            pt_assignment_wrapped=pt_assignment_wrapped,
        )
    )
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
