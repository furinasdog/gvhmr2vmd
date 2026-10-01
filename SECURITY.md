# Security policy / 安全策略

## Supported versions

Security fixes are provided for the latest release on the default branch. Older ZIP releases may
not receive backports.

安全修复面向默认分支上的最新版本，旧 ZIP 不保证回移修复。

## Reporting a vulnerability

Do not publish credentials, remote-host details, private videos, or an exploitable vulnerability
in a public issue. Use GitHub's private vulnerability reporting feature when enabled. If it is not
available, open a minimal issue asking the maintainers for a private contact channel without
including exploit details.

不要在公开 Issue 中发布凭据、远端主机信息、私人视频或可利用漏洞细节。优先使用 GitHub
私密漏洞报告；若仓库未启用，请只提交一个不含漏洞细节的 Issue，请维护者提供私密渠道。

## Security model

- NPZ files are loaded with `allow_pickle=False` and must contain numeric/string arrays only.
- The WebUI binds to localhost by default and rejects unauthenticated non-local binds.
- Uploaded videos and inference outputs remain on the configured host.
- Model files and PMX assets are untrusted inputs; process them only in an environment you control.
- Do not commit `.env`, SSH keys, passwords, model weights, private media, or inference outputs.

- NPZ 使用 `allow_pickle=False`，只接受数值/字符串数组；
- WebUI 默认仅监听本机，并拒绝无认证的非本机监听；
- 上传视频和推理结果保留在配置的主机上；
- 模型文件和 PMX 属于不可信输入，应只在受控环境处理；
- 禁止提交 `.env`、SSH 密钥、密码、权重、私人媒体或推理结果。
