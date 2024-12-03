# customcommand-contrib

本仓库收集一些通用场景的日志易 SPL 自定义指令程序实现。供阅读《日志易检索参考》手册章节后依然不太理解的同仁们参考。

## 自定义命令介绍

### 1. `analyze_sentiment`

**功能：**
- 该脚本用于对流式数据进行情感分析。
- 使用 `pysenti` 库中的 `ModelClassifier` 对文本进行情感分类，判断其为正面或负面。
- 脚本处理数据表中的每一行，添加 `label` 和 `score` 字段以指示情感及其概率。
- 默认分析 input 字段内容。也可以通过 `input=` 参数指定其他字段名。

**安装要求：**
- Python 3
- `pysenti` 库用于情感分析
- 确保脚本中自定义库路径设置正确。

### 2. `dpp`

**功能：**
- 该脚本用于使用行列式点过程（DPP）和语言模型对数据进行处理和总结。
- 处理数据聚类，使用 DPP 进行采样，通过并发请求百度千帆 API 获取每个聚类的采样摘要，再通过阿里云 Dashscope API 生成最终摘要内容。
- 参数 `k` 用于指定每个聚类中要采样的数量，默认值为 5。
- 参数 `by_field` 用于指定输出每个聚类的采样摘要，默认不设置，输出最终摘要内容。

**安装要求：**
- Python 3
- 库：`requests`、`numpy`、`scikit-learn`、`concurrent.futures`
- 确保脚本中自定义库路径设置正确。
- 配置百度千帆和阿里云 Dashscope 的 API 密钥。


## 安装说明

1. **安装所需库：**
   - 日志易 SPL 自定义指令需要再日志易内置 Python 环境中运行。
   - 日志易内置 Python 默认带有 numpy、scikit-learn 等常见机器学习库。
   - 使用日志易内置 Python 安装 `analyze_sentiment` 指令所需的 `pysenti` 库：
   ```bash
   /opt/rizhiyi/parcels/python/bin/python /opt/rizhiyi/parcels/python/bin/pip install pysenti
   ```
   - 也可以自己下载 `pysenti` 库源码后，和指令程序一起打包。

2. **配置 API 密钥：**
   - 在 `dpp.py` 中更新 `QIANFAN_API_KEY`、`QIANFAN_SECRET_KEY` 和 `DASHSCOPE_API_KEY` 为您的 API 凭证。

3. **部署运行：**
   - 将指令对应脚本，通过 `tar zcf XX.tgz XX.py` 命令打包后，上传到日志易 -> 查询分析 -> 指令配置 -> 程序配置列表。
   - 在日志易 -> 查询分析 -> 指令配置页面创建对应的指令配置，请注意选择对应的指令类型。`analyze_sentiment` 是分布式处理命令，`dpp` 是集中处理命令。