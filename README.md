# Torch → MindSpore 转换器（实验性）

使用 json 描述的 API 映射和 LibCST 做语法级改写，将 PyTorch 代码批量转换为 MindSpore 代码。

## 环境依赖
- Python 3.9+
- 安装依赖：`pip install libcst`

## 快速使用
1. 准备好 `api_map.json`（已提供样例）。
2. 运行转换：`python torch2ms.py <源文件.py>`
3. 生成结果：
   - 转换后文件：`<源文件>_ms.py`
   - 差异文件：`diff_(<源文件>-<源文件_ms>).diff`
4. 控制台会打印一次 unified diff，便于快速预览改动。

## 说明
- 转换器根据导入语句推断 MindSpore 前缀（支持 mindspore 与 mindspore.mint）。
- 参数映射遵循 `api_map.json`，缺失或默认值不一致的情况会写在行尾注释中。
- 仍为实验阶段，遇到未覆盖的 API 或语法请自行补充映射或调整代码。 
