"""
torch2ms 主脚本。

读取预置的 PyTorch→MindSpore API 对照表，借助 LibCST 对源码做抽象语法树级改写，
同时生成必要的提示注释、补全 MindSpore 统一导入前缀，并输出转换后的代码与 diff。
"""

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
        use_mint: bool = False,
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
        default_ms_aliases = {
            "mindspore": "ms",
            "mindspore.nn": "msnn",
            "mindspore.ops": "msops",
            "mindspore.mint": "mint",
            "mindspore.mint.nn": "nn",
            "mindspore.mint.ops": "ops",
        }
        self.ms_aliases = default_ms_aliases
        if ms_aliases:
            self.ms_aliases.update(ms_aliases)
        self.pt_aliases = pt_aliases or {}
        self.pt_assignment_aliases = pt_assignment_aliases or {}
        self.pt_assignment_wrapped = pt_assignment_wrapped or {}
        self.use_mint = use_mint
        self.notes_by_stmt = defaultdict(list)
        self.notes_by_func = defaultdict(list)
        self.cell_class_stack = []
        self.api_by_class = {
            conf["pytorch"].split(".")[-1]: conf for conf in api_map.values()
        }
        self.api_by_pt_full = {
            conf["pytorch"]: conf for conf in api_map.values()
        }

        self.torch_dtype_names = {
            "float32",
            "float64",
            "float16",
            "bfloat16",
            "float",
            "double",
            "half",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
            "complex64",
            "complex128",
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

    def _find_enclosing_func(self, node: cst.CSTNode) -> Optional[cst.FunctionDef]:
        """
        向上寻找最近的函数定义节点，便于在函数签名附近添加注释。

        用于收集需要展示在函数声明周围的提示信息，避免行尾注释分散难以查阅。
        """
        parent = self.get_metadata(metadata.ParentNodeProvider, node, None)
        while parent is not None and not isinstance(parent, cst.FunctionDef):
            parent = self.get_metadata(metadata.ParentNodeProvider, parent, None)
        return parent

    def _record_func_note(self, node: cst.CSTNode, note: str) -> None:
        """
        将提示绑定到最近的函数定义，便于批量添加到函数上方。

        适合放置函数级别的提醒（如类型标注缺失映射），在 leave_FunctionDef 阶段统一落盘。
        """
        func = self._find_enclosing_func(node)
        if func:
            self.notes_by_func[func].append(note)

    def _is_ms_cell_base(self, base_name: Optional[str]) -> bool:
        """
        判断基类名是否为 MindSpore Cell（包含常见别名）。
        """
        if not base_name:
            return False
        prefixes = (
            "mindspore.nn.",
            "mindspore.mint.nn.",
            "msnn.",
            "ms.nn.",
            "mint.nn.",
            "nn.",
        )
        return base_name.endswith(".Cell") or base_name in {
            "mindspore.nn.Cell",
            "mindspore.mint.nn.Cell",
            "msnn.Cell",
            "ms.nn.Cell",
            "mint.nn.Cell",
            "nn.Cell",
            "Cell",
        } or any(base_name.startswith(p) and base_name[len(p):] == "Cell" for p in prefixes)

    def _wrap_sequential_args(self, args):
        """
        将 SequentialCell 的可变参数包装成列表，使调用符合 MindSpore 约定。

        - 仅在纯位置参数且未显式提供 list/keyword 时包装；
        - 已是关键字或单列表入参则直接返回；
        - 以多行 list 形式输出以提升可读性。
        """
        if not args:
            return args
        if any(arg.keyword is not None for arg in args):
            return args
        if len(args) == 1 and isinstance(args[0].value, cst.List):
            return args
        # 使用多行列表，尽量保留可读性
        values_code = [
            cst.Module([]).code_for_node(arg.value)
            for arg in args
        ]
        list_code = "[\n" + ",\n".join(f"    {code}" for code in values_code) + "\n]"
        list_expr = cst.parse_expression(list_code)
        return [cst.Arg(keyword=None, value=list_expr)]

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        """
        遍历类定义时记录“是否为 Cell”这一上下文。

        - 若基类来自 torch.nn.Module 及其别名，则标记为需要后续重命名 forward。
        - 若基类已是 MindSpore Cell，也视为需要保持 construct 语义。
        结果会压栈到 `cell_class_stack`，在 leave_FunctionDef 中使用。
        """
        base_names = [
            cst_helpers.get_full_name_for_node(base.value) for base in node.bases
        ]
        is_cell = False
        for name in base_names:
            pt_full_name = self._resolve_pt_full_name(name)
            if pt_full_name and pt_full_name.endswith("Module"):
                is_cell = True
                break
            if self._is_ms_cell_base(name):
                is_cell = True
                break
        self.cell_class_stack.append(is_cell)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        """
        处理函数定义收尾阶段：

        1) 若处于 Cell 类中，自动将 forward 重命名为 construct；
        2) 将此前记录的提示信息插入到函数前的空行注释中，避免丢失人工确认项。
        """
        if self.cell_class_stack and self.cell_class_stack[-1]:
            if updated_node.name.value == "forward":
                updated_node = updated_node.with_changes(name=cst.Name("construct"))

        notes = self.notes_by_func.pop(original_node, [])
        if notes:
            leading = list(updated_node.leading_lines)
            for note in dict.fromkeys(notes):
                leading.append(cst.EmptyLine(comment=cst.Comment(f"# {note}")))
            updated_node = updated_node.with_changes(leading_lines=leading)
        return updated_node

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

        ms_args = []
        mismatch_notes = []

        pos_idx = 0
        force_keyword = False

        vararg_param_names = {"tensors", "inputs"}
        if (
            params
            and params[0]["pytorch"].get("name") in vararg_param_names
            and positional_values
            and (len(positional_values) > 1 or len(positional_values) > len(params))
        ):
            ms_args.extend(cst.Arg(keyword=None, value=val) for val in positional_values)
            for idx, p in enumerate(params[1:], start=1):
                pt = p["pytorch"]
                ms = p["mindspore"]
                pt_name = pt.get("name")
                ms_name = ms.get("name")
                pt_def = pt.get("default")
                ms_def = ms.get("default")
                if pt_name and pt_name in keyword_values:
                    kw_name = ms_name or pt_name
                    if pt_name != ms_name and ms_name is not None:
                        mismatch_notes.append(
                            f"'{api_conf['pytorch']}':默认参数名不一致(position {idx}): PyTorch={pt_name}, MindSpore={ms_name};"
                        )
                    ms_args.append(cst.Arg(keyword=cst.Name(kw_name), value=keyword_values[pt_name]))
                elif (
                    pt_name is not None
                    and ms_name is not None
                    and pt_def is not None
                    and ms_def is not None
                    and pt_def != ms_def
                ):
                    mismatch_notes.append(
                        f"'{api_conf['pytorch']}'默认值不一致(position {idx}): PyTorch={pt_def}, MindSpore={ms_def};"
                    )
            return ms_args, mismatch_notes

        size_like_names = {"size", "shape", "dims"}
        size_bundled = False
        if (
            params
            and params[0]["pytorch"].get("name") in size_like_names
            and params[0]["mindspore"].get("name") in size_like_names
            and positional_values
            and len(positional_values) > 1
            and params[0]["pytorch"].get("name") not in keyword_values
        ):
            size_elems = [cst.Element(value=v) for v in positional_values]
            size_tuple = cst.Tuple(elements=size_elems)
            ms_args.append(
                cst.Arg(keyword=cst.Name(params[0]["mindspore"]["name"]), value=size_tuple)
            )
            size_bundled = True
            pos_idx = len(positional_values)

        for idx, p in enumerate(params):
            pt = p["pytorch"]
            ms = p["mindspore"]
            pt_name = pt.get("name")
            ms_name = ms.get("name")
            pt_def = pt.get("default")
            ms_def = ms.get("default")

            if size_bundled and idx == 0:
                continue

            value = None
            used_keyword = False

            if pt_name and pt_name in keyword_values:
                value = keyword_values[pt_name]
                used_keyword = True
            elif pos_idx < len(positional_values):
                value = positional_values[pos_idx]
                pos_idx += 1

            if value is None:
                if (
                    pt_name is not None
                    and ms_name is not None
                    and pt_def is not None
                    and ms_def is not None
                    and pt_def != ms_def
                ):
                    mismatch_notes.append(
                        f"'{api_conf['pytorch']}'默认值不一致(position {idx}): PyTorch={pt_def}, MindSpore={ms_def};"
                    )
                continue

            if ms_name is None:
                mismatch_notes.append(
                    f"'{api_conf['pytorch']}':没有对应的mindspore参数 '{pt_name}' (position {idx});"
                )
                force_keyword = True
                continue

            if pt_name and pt_name != ms_name:
                mismatch_notes.append(
                    f"'{api_conf['pytorch']}':默认参数名不一致(position {idx}): PyTorch={pt_name}, MindSpore={ms_name};"
                )

            if used_keyword or force_keyword:
                ms_args.append(cst.Arg(keyword=cst.Name(ms_name), value=value))
            else:
                ms_args.append(cst.Arg(keyword=None, value=value))

            if force_keyword is False and used_keyword:
                force_keyword = True

        for extra_val in positional_values[pos_idx:]:
            ms_args.append(cst.Arg(keyword=None, value=extra_val))

        # 确保 LibCST 参数顺序合法：位置参数必须在关键字参数之前，否则会触发
        # "Cannot have positional argument after keyword argument" 的校验错误。
        if any(arg.keyword is None for arg in ms_args) and any(arg.keyword is not None for arg in ms_args):
            positional = [arg for arg in ms_args if arg.keyword is None]
            keyword = [arg for arg in ms_args if arg.keyword is not None]
            ms_args = positional + keyword

        return ms_args, mismatch_notes

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
        missing_recorded = False

        if pt_full_name:
            api_conf = self.api_by_pt_full.get(pt_full_name) or self.api_by_class.get(pt_full_name.split(".")[-1])
            if api_conf:
                new_args, notes = self._reconstruct_args(updated_node, api_conf)
                ms_full_name = self._resolve_ms_full_name(api_conf["mindspore"])
                if api_conf["mindspore"].endswith("SequentialCell"):
                    new_args = self._wrap_sequential_args(new_args)
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
                if api_conf["mindspore"].endswith("SequentialCell"):
                    new_args = self._wrap_sequential_args(new_args)
                new_call = updated_node.with_changes(args=new_args)
                if notes:
                    self._record_note(original_node, notes)
                return new_call
            self._record_note(
                original_node,
                [f"'{wrapper_pt_full}' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;"],
            )
            missing_recorded = True

        if pt_full_name and not missing_recorded:
            self._record_note(
                original_node,
                [f"'{pt_full_name}' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;"],
            )
        return updated_node

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        """
        拦截属性访问，统一将 `torch.float32` 等 dtype 名替换为 `ms.float32`。

        仅处理以 torch 开头、且后缀在 `torch_dtype_names` 列表内的情况，其他属性保持不变。
        """
        full_name = cst_helpers.get_full_name_for_node(updated_node)
        if not full_name:
            return updated_node
        pt_full_name = self._resolve_pt_full_name(full_name)
        if not pt_full_name or not pt_full_name.startswith("torch."):
            return updated_node
        suffix = pt_full_name[len("torch.") :]
        if suffix in self.torch_dtype_names:
            return cst.parse_expression(f"ms.{suffix}")
        return updated_node

    def leave_Annotation(self, original_node: cst.Annotation, updated_node: cst.Annotation) -> cst.Annotation:
        """
        处理类型标注中的 torch.*，优先按映射表替换；缺失时记录提示。
        """
        full_name = cst_helpers.get_full_name_for_node(updated_node.annotation)
        pt_full_name = self._resolve_pt_full_name(full_name)
        if not pt_full_name:
            return updated_node

        api_conf = self.api_by_pt_full.get(pt_full_name) or self.api_by_class.get(
            pt_full_name.split(".")[-1]
        )
        if api_conf:
            ms_full_name = self._resolve_ms_full_name(api_conf["mindspore"])
            new_expr = cst.parse_expression(ms_full_name)
            return updated_node.with_changes(annotation=new_expr)

        self._record_func_note(
            original_node,
            f"类型标注 '{pt_full_name}' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;",
        )
        return updated_node

    def leave_ClassDef(
        self,
        original_node: cst.ClassDef,
        updated_node: cst.ClassDef,
    ) -> cst.ClassDef:
        """
        将继承自 torch.nn.Module 等基类的类，尝试替换为 MindSpore 的对应基类。

        示例:
            >>> code = "import torch.nn as nn\\nclass M(nn.Module):\\n    pass"
            >>> mod = cst.parse_module(code)
            >>> wrapper = metadata.MetadataWrapper(mod)
            >>> new_mod = wrapper.visit(TorchToMindSporeTransformer(API_MAP, pt_aliases={'nn': 'torch.nn'}))
            >>> "mindspore" in new_mod.code
            True

        输出:
            若能根据别名还原出 torch 全路径，并在映射表中找到对应 MindSpore 基类，则替换之。
        """
        if self.cell_class_stack:
            self.cell_class_stack.pop()

        if not updated_node.bases:
            return updated_node

        new_bases = []
        changed = False
        missing_notes = []
        for base in updated_node.bases:
            full_name = cst_helpers.get_full_name_for_node(base.value)
            pt_full_name = self._resolve_pt_full_name(full_name)
            ms_expr = None

            if pt_full_name:
                api_conf = self.api_by_pt_full.get(pt_full_name) or self.api_by_class.get(
                    pt_full_name.split(".")[-1]
                )
                if api_conf:
                    ms_full_name = self._resolve_ms_full_name(api_conf["mindspore"])
                    ms_expr = cst.parse_expression(ms_full_name)
                elif pt_full_name == "torch.nn.Module":
                    # 兜底逻辑: 未在映射表中配置时，默认映射到 mindspore.nn.Cell
                    ms_full_name = self._resolve_ms_full_name("mindspore.nn.Cell")
                    ms_expr = cst.parse_expression(ms_full_name)
                else:
                    missing_notes.append(f"'{pt_full_name}' 未在映射表(api_mapping_out_excel.json)中找到，需手动确认;")

            if ms_expr is not None:
                new_bases.append(base.with_changes(value=ms_expr))
                changed = True
            else:
                new_bases.append(base)

        if missing_notes:
            leading = list(updated_node.leading_lines)
            for note in dict.fromkeys(missing_notes):
                leading.append(cst.EmptyLine(comment=cst.Comment(f"# {note}")))
            updated_node = updated_node.with_changes(leading_lines=leading)

        if changed:
            return updated_node.with_changes(bases=new_bases)
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


class _NameUsageCollector(cst.CSTVisitor):
    """
    收集除导入语句以外的位置上出现的所有标识符名称，用于判断某个 import 是否仍然被使用。
    """

    def __init__(self) -> None:
        """
        初始化名称收集器，准备一个空集合来记录实际被引用的标识符。
        """
        self.used_names = set()

    def visit_Import(self, node: cst.Import) -> None:
        """
        避免进入 import 语句内部，直接返回 False 让遍历跳过子节点。

        这样可以防止把“别名定义本身”视作使用行为，只统计后续真实引用。
        """
        # 不进入 import 语句内部，避免把别名本身也统计为“使用”
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        """
        与 `visit_Import` 相同，跳过 from ... import ... 的内部遍历。
        """
        # 同上，跳过 from ... import ... 的内部
        return False

    def visit_Name(self, node: cst.Name) -> None:
        """
        收集实际出现的标识符名称，供后续导入清理判断是否仍被使用。
        """
        self.used_names.add(node.value)


class TorchImportCleaner(cst.CSTTransformer):
    """
    根据名称使用情况，移除不再需要的 torch 相关导入。

    仅当某个 import 里所有别名均未在后续代码中使用时，才安全地删除该 import。
    """

    def __init__(self, used_names) -> None:
        """
        保存一次遍历后得到的“已使用名称”集合，供 import 清理阶段参考。
        """
        super().__init__()
        self.used_names = set(used_names)

    def leave_Import(self, original_node: cst.Import, updated_node: cst.Import) -> cst.BaseStatement:
        """
        清理普通 import 中未被使用的 torch 别名；若整条语句均无用则删除。
        """
        new_aliases = []
        for alias in updated_node.names:
            module = cst_helpers.get_full_name_for_node(alias.name)
            alias_name = (
                alias.asname.name.value
                if getattr(alias, "asname", None)
                else getattr(alias, "evaluated_name", None)
            ) or module

            if not module or not module.startswith("torch"):
                new_aliases.append(alias)
                continue

            if alias_name in self.used_names:
                new_aliases.append(alias)

        if not new_aliases:
            return cst.RemoveFromParent()
        return updated_node.with_changes(names=new_aliases)

    def leave_ImportFrom(
        self,
        original_node: cst.ImportFrom,
        updated_node: cst.ImportFrom,
    ) -> cst.BaseStatement:
        """
        清理 from ... import ... 语句下未使用的 torch 别名，星号导入保守保留。
        """
        if updated_node.module is None:
            return updated_node

        module_name = cst_helpers.get_full_name_for_node(updated_node.module)
        if not module_name or not module_name.startswith("torch"):
            return updated_node

        new_names = []
        for name in updated_node.names:
            if isinstance(name, cst.ImportStar):
                # 星号导入保守保留
                new_names.append(name)
                continue

            alias_name = (
                name.asname.name.value
                if getattr(name, "asname", None)
                else getattr(name, "evaluated_name", None)
            ) or name.name.value

            if alias_name in self.used_names:
                new_names.append(name)

        if not new_names:
            return cst.RemoveFromParent()
        return updated_node.with_changes(names=tuple(new_names))


def _ensure_default_ms_imports(code: str) -> str:
    """
    确保转换后的代码包含统一的 mindspore 导入前缀。

    若缺失标准导入(ms/msnn/msops/mint 及 from mint import nn, ops)，则在文件头部插入；
    已存在则不重复写入，保证幂等。
    """
    required = [
        "import mindspore as ms\n",
        "import mindspore.nn as msnn\n",
        "import mindspore.ops as msops\n",
        "import mindspore.mint as mint\n",
        "from mindspore.mint import nn, ops\n",
    ]
    lines = code.splitlines(keepends=True)
    existing = {line.strip() for line in lines}
    missing = [line for line in required if line.strip() not in existing]
    if not missing:
        return code

    insert_idx = 0
    if lines and lines[0].startswith("#!"):
        insert_idx = 1
    if len(lines) > insert_idx and "coding" in lines[insert_idx]:
        insert_idx += 1

    for line in reversed(missing):
        lines.insert(insert_idx, line)

    return "".join(lines)


def _comment_torch_imports(code: str) -> str:
    """
    注释掉包含 torch 的 import 行，保留缩进。

    在清理无用导入后调用，对仍残留的 torch 导入做“软屏蔽”，方便人工审阅和
    逐步迁移，同时避免破坏原有缩进与换行格式。
    """
    new_lines = []
    for line in code.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            new_lines.append(line)
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            if "torch" in stripped:
                indent_len = len(line) - len(stripped)
                indent = line[:indent_len]
                commented = f"{indent}# {stripped}"
                if not commented.endswith("\n") and line.endswith("\n"):
                    commented += "\n"
                new_lines.append(commented)
                continue
        new_lines.append(line)
    return "".join(new_lines)


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

    流程概览:
        1) 收集 torch 导入/赋值别名；
        2) 依据映射表改写调用和类型标注，并记录提示；
        3) 清理已失效的 torch 导入，剩余的加注释屏蔽；
        4) 补齐标准化的 MindSpore 导入前缀。
    """
    module = cst.parse_module(code)
    api_by_class = {
        conf["pytorch"].split(".")[-1]: conf for conf in API_MAP.values()
    }
    pt_aliases = collect_torch_aliases(module)
    pt_assignment_aliases, pt_assignment_wrapped = collect_torch_assignment_aliases(
        module, api_by_class, pt_aliases
    )

    wrapper = metadata.MetadataWrapper(module)
    new_module = wrapper.visit(
        TorchToMindSporeTransformer(
            API_MAP,
            ms_aliases=None,
            pt_aliases=pt_aliases,
            pt_assignment_aliases=pt_assignment_aliases,
            pt_assignment_wrapped=pt_assignment_wrapped,
            use_mint=False,
        )
    )
    # 第二遍遍历: 基于名称使用情况，清理不再需要的 torch 导入
    name_collector = _NameUsageCollector()
    new_module.visit(name_collector)
    cleaner = TorchImportCleaner(name_collector.used_names)
    new_module = new_module.visit(cleaner)

    new_code = new_module.code
    new_code = _comment_torch_imports(new_code)
    new_code = _ensure_default_ms_imports(new_code)

    return new_code


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


def _convert_and_save(filename: str, input_root: Optional[str] = None, show_diff: bool = True) -> None:
    """
    转换单个文件并写入 output 目录，同时生成 diff。

    当传入 input_root 时，会在 output 下保留相对目录结构:
        input_root/foo/bar.py -> output/<input_root_basename>/foo/bar.py
    否则:
        some.py -> output/some.py

    另外会在 diff/<input_root>_diff 目录按相对路径保存对应的 diff，便于批量审阅。
    """
    with open(filename, "r", encoding="utf8") as f:
        code = f.read()

    result = convert_code(code)

    base, ext = os.path.splitext(filename)
    if input_root:
        abs_input = os.path.abspath(input_root)
        abs_file = os.path.abspath(filename)
        try:
            common = os.path.commonpath([abs_input, abs_file])
        except ValueError:
            common = ""

        # 在 output 下创建“同名目录+_ms”，比如:
        #   input_root = "input"      -> output/input_ms/...
        #   input_root = "vit_pytorch" -> output/vit_pytorch_ms/...
        root_name = os.path.basename(os.path.normpath(abs_input))
        output_root = os.path.join("output", f"{root_name}")

        if common == abs_input:
            rel_base = os.path.relpath(base, abs_input)
            out_base = os.path.join(output_root, rel_base)
        else:
            # 不在 input_root 子树下时退化为: output/<root_name>/<basename>
            out_base = os.path.join(output_root, os.path.basename(base))
    else:
        # 单文件模式: 仍然是 output/xxx_ms.py
        out_base = os.path.join("output", os.path.basename(base))

    new_filename = f"{out_base}{ext}"
    out_dir = os.path.dirname(new_filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(new_filename, "w", encoding="utf8") as f:
        f.write(result)

    diff = generate_diff(code, result)
    if show_diff:
        print("=== 转换 DIFF 开始 ===")
        print(diff)
        print("=== 转换 DIFF 结束 ===")

    # 按 input 目录层级创建 diff 存储路径: diff/<input_root>_diff/<relative>.diff
    if input_root and common == abs_input:
        rel_path = os.path.relpath(filename, abs_input)
        diff_root = os.path.join("diff", f"{os.path.basename(abs_input)}_diff")
        diff_path = os.path.join(diff_root, rel_path + ".diff")
    elif input_root:
        diff_root = os.path.join("diff", f"{os.path.basename(abs_input)}_diff")
        diff_path = os.path.join(diff_root, os.path.basename(filename) + ".diff")
    else:
        diff_root = "diff"
        diff_path = os.path.join(diff_root, f"{os.path.basename(filename)}.diff")

    os.makedirs(os.path.dirname(diff_path), exist_ok=True)
    with open(diff_path, "w", encoding="utf8") as f:
        f.write(diff)
    if show_diff:
        print(f"已保存 diff 到: {diff_path}")

    print(f"已生成新文件: {new_filename}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python torch2ms.py input.py")
        print("  python torch2ms.py input_dir   # 批量转换目录下所有 .py 文件")
        sys.exit(0)

    target = sys.argv[1]

    if os.path.isdir(target):
        print(f"检测到目录，开始批量转换: {target}")
        for root, _, files in os.walk(target):
            for name in files:
                if not name.endswith(".py"):
                    continue
                src_path = os.path.join(root, name)
                print(f"\n[文件] {src_path}")
                _convert_and_save(src_path, input_root=target, show_diff=False)
        print("\n批量转换完成。")
    else:
        abs_target = os.path.abspath(target)
        tp_root = os.path.abspath(os.path.join("input", "torch_third_party"))
        default_root = tp_root if os.path.commonpath([abs_target, tp_root]) == tp_root else None
        _convert_and_save(target, input_root=default_root, show_diff=True)
