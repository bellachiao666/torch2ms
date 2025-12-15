# Torch → MindSpore 转换器（实验性）

使用 json 描述的 API 映射和 LibCST 做语法级改写，将 PyTorch 代码批量转换为 MindSpore 代码。

## 环境依赖
- Python 3.9+
- 安装依赖：`pip install libcst`

## 快速使用
1. 将待转换的 PyTorch 代码放到 `input/torch_third_party/` 下（目录结构可自行扩展、文件名保持不变）。
2. 运行转换：
   - 转换整个三方目录：`python torch2ms.py input/torch_third_party`
   - 或转换单个文件：`python torch2ms.py input/torch_third_party/<模块>/<文件>.py`
3. 生成结果：
   - 转换后文件：落在 `output/torch_third_party/` 下，与 `input/torch_third_party/` 目录层级一一对应，文件名保持不变。
   - 差异文件：保存到 `diff/torch_third_party_diff/`，按相对路径生成 `.diff`。
4. 控制台会打印一次 unified diff，便于快速预览改动。

## 说明
- 转换器根据导入语句推断 MindSpore 前缀（支持 mindspore 与 mindspore.mint）。
- 参数映射遵循 `api_mapping_out_excel.json`，缺失或默认值不一致的情况会写在行尾注释中。
- 仍为实验阶段，遇到未覆盖的 API 或语法请自行补充映射或调整代码。 
