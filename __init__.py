"""pkg - 聚合封装包

本包以磁盘上以数字前缀命名的模块文件为来源（例如 `1_image_transforms.py`），
在导入时把这些模块按去掉数字前缀后的名字暴露为包属性，方便使用：

from pkg import image_transforms  -> 加载自 1_image_transforms.py
"""
from pathlib import Path
import importlib.util
import sys
import re

_HERE = Path(__file__).parent

__all__ = []

for p in sorted(_HERE.glob("*.py")):
    if p.name == "__init__.py":
        continue
    stem = p.stem
    # 去掉前缀数字和下划线，例如 1_image_transforms -> image_transforms
    name = re.sub(r"^\d+_", "", stem)

    spec = importlib.util.spec_from_file_location(f"pkg._{stem}", str(p))
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        continue
    try:
        loader.exec_module(module)
    except Exception as e:
        # 如果模块依赖缺失或运行时代码抛出异常，不要中断包导入，记录模块对象但附带错误信息属性
        setattr(module, "__load_error__", e)
    # 将模块对象以去掉数字前缀的名字暴露在包命名空间
    globals()[name] = module
    # 将模块注册为包的子模块，方便使用 from pkg.<name> import ...
    try:
        sys.modules[f"pkg.{name}"] = module
    except Exception:
        pass
    __all__.append(name)

del Path, importlib, sys, re, _HERE, p, stem, name, spec, module, loader
