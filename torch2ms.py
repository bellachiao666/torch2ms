import difflib
import json
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Optional

import libcst as cst
from libcst import helpers as cst_helpers
from libcst import metadata


with open("api_mapping_out_excel.json", "r", encoding="utf8") as f:
    API_MAP = json.load(f)["apis"]


def _literal_to_expr(value):
    """
    将 Python 字面量转成 LibCST 表达式节点，便于后续直接拼装到 AST 中。

    示例:
        >>> expr = _literal_to_expr({"flag": True, "items": [1, 2]})
        >>> isinstance(expr, cst.BaseExpression)
        True
        >>> expr.dump()
        "Dict(elements=[Element(key=SimpleString(value='\"flag\"'), value=Name(value='\"True\"')), ...])"

    输出:
        返回与输入值结构一致的 LibCST 表达式节点，`dump()` 后可看到完整结构。
    """
    return cst.parse_expression(repr(value))


class _MindSporeImportCollector(cst.CSTVisitor):
    """
    读取导入语句，记录所有 MindSpore 模块前缀（不限于 nn）。

    示例:
        >>> code = "import mindspore.nn as mnn\\nfrom mindspore import ops"
        >>> collector = _MindSporeImportCollector()
        >>> cst.parse_module(code).visit(collector)
        >>> collector.alias_by_module
        {'mindspore.nn': 'mnn', 'mindspore.ops': 'mindspore.ops'}

    输出:
        访问导入节点后在 `alias_by_module` 中得到模块到别名的映射。
    """

    def __init__(self) -> None:
        """
        初始化 MindSpore 导入收集器，准备记录模块别名。

        示例:
            >>> collector = _MindSporeImportCollector()
            >>> collector.alias_by_module
            {}

        输出:
            生成一个空的 `alias_by_module` 字典，等待填充。
        """
        self.alias_by_module = {}

    def _maybe_record(self, module: str, alias: str) -> None:
        """
        判断并记录符合前缀的模块别名。

        示例:
            >>> collector = _MindSporeImportCollector()
            >>> collector._maybe_record("mindspore.nn", "mnn")
            >>> collector.alias_by_module
            {'mindspore.nn': 'mnn'}

        输出:
            当模块前缀以 mindspore 开头时，将别名写入 `alias_by_module`。
        """
        if not module:
            return
        if not module.startswith("mindspore"):
            return
        self.alias_by_module[module] = alias

    def visit_Import(self, node: cst.Import) -> None:
        """
        处理 `import` 语句收集 MindSpore 别名。

        示例:
            >>> code = "import mindspore.nn as mnn"
            >>> collector = _MindSporeImportCollector()
            >>> cst.parse_module(code).visit(collector)
            >>> collector.alias_by_module
            {'mindspore.nn': 'mnn'}

        输出:
            导入语句访问后，别名被记录到 `alias_by_module`。
        """
        for name in node.names:
            module = cst_helpers.get_full_name_for_node(name.name)
            alias = (
                name.asname.name.value
                if getattr(name, "asname", None)
                else getattr(name, "evaluated_name", None)
            ) or module
            self._maybe_record(module, alias)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """
        处理 `from ... import ...` 语句，补充 MindSpore 模块别名。

        示例:
            >>> code = "from mindspore import ops"
            >>> collector = _MindSporeImportCollector()
            >>> cst.parse_module(code).visit(collector)
            >>> collector.alias_by_module
            {'mindspore.ops': 'ops'}

        输出:
            针对非星号导入，将完整模块路径与别名的对应关系写入。
        """
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
    根据导入语句收集 MindSpore 模块前缀映射，便于后续还原完整调用名。

    示例:
        >>> mod = cst.parse_module("import mindspore.nn as mnn\\nfrom mindspore import ops")
        >>> collect_mindspore_aliases(mod)
        {'mindspore.nn': 'mnn', 'mindspore.ops': 'ops'}

    输出:
        返回字典，键为 MindSpore 模块全名，值为代码中的别名。
    """
    collector = _MindSporeImportCollector()
    module.visit(collector)
    return collector.alias_by_module


class _TorchImportCollector(cst.CSTVisitor):
    """
    收集 torch 模块别名，记录 alias -> module 完整路径。

    示例:
        >>> code = "import torch.nn as tn\\nfrom torch import optim"
        >>> collector = _TorchImportCollector()
        >>> cst.parse_module(code).visit(collector)
        >>> collector.module_by_alias
        {'tn': 'torch.nn', 'optim': 'torch.optim'}

    输出:
        完成遍历后 `module_by_alias` 保存 torch 别名映射。
    """

    def __init__(self) -> None:
        """
        初始化 torch 导入收集器。

        示例:
            >>> _TorchImportCollector().module_by_alias
            {}

        输出:
            创建空的别名记录字典。
        """
        self.module_by_alias = {}

    def _maybe_record(self, module: str, alias: str) -> None:
        """
        判断模块是否 torch 前缀并记录别名。

        示例:
            >>> collector = _TorchImportCollector()
            >>> collector._maybe_record("torch.nn", "tn")
            >>> collector.module_by_alias
            {'tn': 'torch.nn'}

        输出:
            当模块以 torch 开头时写入 `module_by_alias`。
        """
        if not module or not module.startswith("torch"):
            return
        self.module_by_alias[alias] = module

    def visit_Import(self, node: cst.Import) -> None:
        """
        处理 `import` 语句，将 torch 导入记录到别名表。

        示例:
            >>> code = "import torch as T"
            >>> collector = _TorchImportCollector()
            >>> cst.parse_module(code).visit(collector)
            >>> collector.module_by_alias
            {'T': 'torch'}

        输出:
            别名映射中包含 import 语句的别名关系。
        """
        for name in node.names:
            module = cst_helpers.get_full_name_for_node(name.name)
            alias = (
                name.asname.name.value
                if getattr(name, "asname", None)
                else getattr(name, "evaluated_name", None)
            ) or module
            self._maybe_record(module, alias)

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """
        处理 `from ... import ...` 语句，将 torch 下的导入也映射成完整模块。

        示例:
            >>> code = "from torch import nn"
            >>> collector = _TorchImportCollector()
            >>> cst.parse_module(code).visit(collector)
            >>> collector.module_by_alias
            {'nn': 'torch.nn'}

        输出:
            将 from 导入的名称补全成 torch 的全路径后写入映射。
        """
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
    """
    根据导入语句收集 torch 模块别名，返回 alias -> 全路径 的映射。

    示例:
        >>> mod = cst.parse_module("import torch.nn as tn\\nfrom torch import optim")
        >>> collect_torch_aliases(mod)
        {'tn': 'torch.nn', 'optim': 'torch.optim'}

    输出:
        字典形式的别名映射，用于后续还原完整 PyTorch API 名称。
    """
    collector = _TorchImportCollector()
    module.visit(collector)
    return collector.module_by_alias


class _TorchAliasAssignmentCollector(cst.CSTVisitor):
    """
    记录简单赋值形成的别名映射，例如: `myconv = nn.Conv2d`。

    示例:
        >>> code = "import torch.nn as nn\\nConv = nn.Conv2d"
        >>> collector = _TorchAliasAssignmentCollector(api_by_class={}, pt_aliases={"nn": "torch.nn"})
        >>> cst.parse_module(code).visit(collector)
        >>> collector.alias_to_pt
        {'Conv': 'torch.nn.Conv2d'}

    输出:
        遍历赋值后，`alias_to_pt` 保存直接别名，`alias_to_pt_wrapped` 保存包装类对应的原始 PyTorch API。
    """

    def __init__(self, api_by_class: dict, pt_aliases: dict) -> None:
        """
        初始化赋值别名收集器，依赖已有的 API 配置与 torch 别名。

        示例:
            >>> _TorchAliasAssignmentCollector({"Conv2d": {"pytorch": "torch.nn.Conv2d"}}, {"nn": "torch.nn"})
            <_TorchAliasAssignmentCollector ...>

        输出:
            内部别名映射被置空，等待遍历代码时填充。
        """
        self.api_by_class = api_by_class
        self.pt_aliases = pt_aliases
        self.alias_to_pt = {}
        self.alias_to_pt_wrapped = {}


    def _resolve_pt_full_name(self, full_name: Optional[str]) -> Optional[str]:
        """
        根据别名映射还原 PyTorch 调用的完整路径。

        示例:
            >>> collector = _TorchAliasAssignmentCollector({}, {"nn": "torch.nn"})
            >>> collector._resolve_pt_full_name("nn.Linear")
            'torch.nn.Linear'

        输出:
            如果能找到匹配的前缀则返回完整路径，否则返回 None。
        """
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
        """
        处理单个赋值节点，记录左侧别名对应的 PyTorch API。

        示例:
            >>> collector = _TorchAliasAssignmentCollector({"Conv2d": {"pytorch": "torch.nn.Conv2d"}}, {"nn": "torch.nn"})
            >>> assign = cst.parse_statement("my_conv = nn.Conv2d").body[0]
            >>> collector._maybe_record_alias(assign.targets[0].target, assign.value)
            >>> collector.alias_to_pt
            {'my_conv': 'torch.nn.Conv2d'}

        输出:
            更新 `alias_to_pt` 或 `alias_to_pt_wrapped`，便于后续替换调用。
        """
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
        """
        捕获简单赋值语句并委托记录别名。

        示例:
            >>> collector = _TorchAliasAssignmentCollector({"Linear": {"pytorch": "torch.nn.Linear"}}, {"nn": "torch.nn"})
            >>> cst.parse_module("layer = nn.Linear").visit(collector)
            >>> collector.alias_to_pt
            {'layer': 'torch.nn.Linear'}

        输出:
            将赋值中发现的别名写入映射。
        """
        if len(node.targets) != 1:
            return
        self._maybe_record_alias(node.targets[0].target, node.value)

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        """
        捕获带类型标注的赋值并记录别名。

        示例:
            >>> collector = _TorchAliasAssignmentCollector({"Conv2d": {"pytorch": "torch.nn.Conv2d"}}, {"nn": "torch.nn"})
            >>> cst.parse_module("conv: Any = nn.Conv2d").visit(collector)
            >>> collector.alias_to_pt
            {'conv': 'torch.nn.Conv2d'}

        输出:
            即使存在类型标注，也会保留别名映射信息。
        """
        self._maybe_record_alias(node.target, node.value)


def collect_torch_assignment_aliases(module: cst.Module, api_by_class: dict, pt_aliases: dict) -> dict:
    """
    收集代码中通过赋值形成的别名，返回直接别名和包装别名的双重映射。

    示例:
        >>> mod = cst.parse_module("import torch.nn as nn\\nConv = nn.Conv2d")
        >>> collect_torch_assignment_aliases(mod, {"Conv2d": {"pytorch": "torch.nn.Conv2d"}}, {"nn": "torch.nn"})
        ({'Conv': 'torch.nn.Conv2d'}, {})

    输出:
        一个二元组: (alias_to_pt, alias_to_pt_wrapped)。
    """
    collector = _TorchAliasAssignmentCollector(api_by_class, pt_aliases)
    module.visit(collector)
    return collector.alias_to_pt, collector.alias_to_pt_wrapped


class TorchToMindSporeTransformer(cst.CSTTransformer):
    """
    使用 LibCST 将 PyTorch API 调用改写为 MindSpore 调用。

    示例:
        >>> code = "import torch.nn as nn\\nx = nn.ReLU()(input)"
        >>> mod = cst.parse_module(code)
        >>> wrapper = metadata.MetadataWrapper(mod)
        >>> new_mod = wrapper.visit(TorchToMindSporeTransformer(API_MAP, pt_aliases={'nn': 'torch.nn'}))
        >>> "mindspore" in new_mod.code
        True

    输出:
        生成的新 AST 在 `code` 属性中包含 MindSpore API 调用及必要的注释。
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
        """
        保存 PyTorch->MindSpore 的映射及收集到的别名信息。

        示例:
            >>> transformer = TorchToMindSporeTransformer(API_MAP, ms_aliases={'mindspore.nn': 'ms.nn'}, pt_aliases={'nn': 'torch.nn'})
            >>> transformer.ms_aliases
            {'mindspore.nn': 'ms.nn'}

        输出:
            Transformer 持有转换所需的别名和参数配置，准备在遍历时使用。
        """
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
        """
        获取包裹当前节点的最外层简单语句节点，方便追加注释。

        示例:
            >>> wrapper = metadata.MetadataWrapper(cst.parse_module("x = fn()"))
            >>> transformer = TorchToMindSporeTransformer(API_MAP)
            >>> wrapper.visit(transformer)  # 让 transformer 拥有父节点元数据
            >>> expr_node = wrapper.tree.body[0].body[0].value
            >>> isinstance(transformer._find_enclosing_stmt(expr_node), cst.SimpleStatementLine)
            True

        输出:
            返回 `cst.SimpleStatementLine` 或 None（未找到父节点时）。
        """
        parent = self.get_metadata(metadata.ParentNodeProvider, node, None)
        while parent is not None and not isinstance(parent, cst.SimpleStatementLine):
            parent = self.get_metadata(metadata.ParentNodeProvider, parent, None)
        return parent

    def _record_note(self, node: cst.CSTNode, notes) -> None:
        """
        将参数差异等提示记录到语句末尾，方便人工确认。

        示例:
            >>> wrapper = metadata.MetadataWrapper(cst.parse_module("y = fn()"))
            >>> transformer = TorchToMindSporeTransformer(API_MAP)
            >>> wrapper.visit(transformer)
            >>> stmt = wrapper.tree.body[0]
            >>> transformer._record_note(stmt, ["需要确认默认值"])
            >>> transformer.notes_by_stmt[stmt]
            ['需要确认默认值']

        输出:
            `notes_by_stmt` 被填充，后续在 leave_SimpleStatementLine 中会转成行尾注释。
        """
        stmt = self._find_enclosing_stmt(node)
        if stmt is not None:
            self.notes_by_stmt[stmt].extend(notes)

    def _resolve_ms_full_name(self, full_path: str) -> str:
        """
        使用导入的别名还原 MindSpore API 全路径。

        示例:
            >>> transformer = TorchToMindSporeTransformer(API_MAP, ms_aliases={'mindspore.nn': 'ms.nn'})
            >>> transformer._resolve_ms_full_name("mindspore.nn.ReLU")
            'ms.nn.ReLU'

        输出:
            返回替换了别名后的完整 MindSpore 名称。
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

        示例:
            >>> transformer = TorchToMindSporeTransformer(API_MAP, pt_aliases={'nn': 'torch.nn'})
            >>> transformer._resolve_pt_full_name("nn.Linear")
            'torch.nn.Linear'

        输出:
            成功匹配时返回以 torch 开头的完整路径，否则返回 None。
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
        """
        按 API 映射重排、重命名参数，返回新的实参列表与需要提示的备注。

        示例:
            >>> call = cst.parse_expression("fn(1, stride=2)")
            >>> api_conf = {
            ...     "pytorch": "torch.nn.Conv2d",
            ...     "mindspore": "mindspore.nn.Conv2d",
            ...     "params": [
            ...         {"pytorch": {"name": "in_channels", "default": None}, "mindspore": {"name": "in_channels", "default": None}},
            ...         {"pytorch": {"name": "stride", "default": 1}, "mindspore": {"name": "stride", "default": 1}},
            ...     ],
            ... }
            >>> new_args, notes = TorchToMindSporeTransformer(API_MAP)._reconstruct_args(call, api_conf)
            >>> [a.keyword.value for a in new_args]
            ['in_channels', 'stride']
            >>> notes
            []

        输出:
            元组 `(args, notes)`，其中 `args` 为适配 MindSpore 的参数列表，`notes` 为需要人工确认的提示。
        """
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
        """
        在访问完函数/方法调用后，根据映射将 PyTorch 调用替换为 MindSpore 并调整参数。

        示例:
            >>> api_conf = {"pytorch": "torch.nn.ReLU", "mindspore": "mindspore.nn.ReLU", "params": []}
            >>> transformer = TorchToMindSporeTransformer({"torch.nn.ReLU": api_conf}, pt_aliases={"nn": "torch.nn"})
            >>> call = cst.parse_expression("nn.ReLU(x)")
            >>> new_call = transformer.leave_Call(call, call)
            >>> cst_helpers.get_full_name_for_node(new_call.func)
            'mindspore.nn.ReLU'
            True

        输出:
            如果匹配到映射则返回替换后的调用节点，并记录参数差异备注；否则原样返回。
        """
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
        """
        在离开一行简单语句时，若有记录的提示信息则追加为行尾注释。

        示例:
            >>> stmt = cst.parse_statement("x = 1")
            >>> transformer = TorchToMindSporeTransformer(API_MAP)
            >>> transformer.notes_by_stmt[stmt] = ["默认值不一致"]
            >>> new_stmt = transformer.leave_SimpleStatementLine(stmt, stmt)
            >>> new_stmt.trailing_whitespace.comment.value
            '# 默认值不一致'

        输出:
            返回新增行尾注释后的语句节点，若无提示则保持不变。
        """
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

    示例:
        >>> src = "import torch.nn as nn\\nnet = nn.ReLU()"
        >>> result = convert_code(src)
        >>> "mindspore" in result
        True

    输出:
        返回转换后的 MindSpore 源码字符串，包含必要的参数名替换与行尾提示。
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

    示例:
        >>> old = "a = 1\\n"
        >>> new = "a = 2\\n"
        >>> print(generate_diff(old, new))
        --- pytorch
        +++ mindspore
        @@
        -a = 1
        +a = 2

    输出:
        返回标准 unified diff 文本，可直接写入 .diff 文件或打印。
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

    filename = "./input/" + sys.argv[1]

    with open(filename, "r", encoding="utf8") as f:
        code = f.read()

    result = convert_code(code)

    base, ext = os.path.splitext(filename)
    new_filename = f"./output/{os.path.basename(base)}_ms{ext}"

    with open(new_filename, "w", encoding="utf8") as f:
        f.write(result)

    diff = generate_diff(code, result)
    print("=== 转换 DIFF 开始 ===")
    print(diff)
    print("=== 转换 DIFF 结束 ===")

    diff_filename = f"diff_({os.path.basename(filename)}-{os.path.basename(new_filename)}).diff"
    diff_path = os.path.join("./diff", diff_filename)
    with open(diff_path, "w", encoding="utf8") as f:
        f.write(diff)
    print(f"已保存 diff 到: {diff_path}")

    print(f"\n已生成新文件: {new_filename}")
