# Contributing / 参与贡献

Thank you for helping improve GVHMR to MMD Bridge. 中文说明见后半部分。

## English

Before opening an issue:

1. use the latest release;
2. read the troubleshooting section in `docs/USER_GUIDE.en.md`;
3. verify whether the problem belongs to this bridge, GVHMR, Blender, or mmd_tools;
4. remove private paths, credentials, identifiable video frames, and restricted assets.

Bug reports should include Blender and add-on versions, operating system, NPZ format version,
the full error text, recognized/missing bone names, and minimal reproduction steps. Do not upload
PMX models or checkpoints unless their license explicitly permits redistribution.

Pull requests should:

- stay focused and preserve NPZ v1 compatibility unless a documented breaking change is agreed;
- keep Blender-independent math in `core.py` where possible;
- add tests for new mapping names, NPZ fields, or coordinate transforms;
- update both manuals when user-visible behavior changes;
- run the checks below and avoid generated files.

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
ruff check .
python -m compileall -q blender_addon server tools tests
python tools/package_addon.py
```

By submitting a contribution, you agree that it is licensed under GPL-3.0-or-later and that you
have the right to submit it.

## 中文

提交 Issue 前：

1. 使用最新版本；
2. 阅读 `docs/USER_GUIDE.zh-CN.md` 的排错章节；
3. 判断问题属于本桥接项目、GVHMR、Blender 还是 mmd_tools；
4. 删除隐私路径、凭据、可识别人物画面和受限素材。

错误报告应包含 Blender/插件版本、操作系统、NPZ 格式版本、完整报错、识别/缺失骨骼名
和最小复现步骤。除非许可明确允许再分发，否则不要上传 PMX 或检查点。

Pull Request 应保持目标单一；除非已讨论破坏性变更，否则保留 NPZ v1 兼容；为新增映射、
字段和坐标转换添加测试；同步更新中英文手册；提交前运行上述检查。提交贡献即表示你有权
提交相关内容，并同意其采用 GPL-3.0-or-later。
