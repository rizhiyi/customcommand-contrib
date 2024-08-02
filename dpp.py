#!python3
import os
import sys
import requests
import json
import time
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity

executehome = "/opt/rizhiyi/parcels/splserver"
lib_path = os.path.join(executehome, 'bin', 'custom_commands', 'lib')
sys.path.insert(0, lib_path)

from src.cn.yottabyte.process.centralized_handler import CentralizedHandler
from src.cn.yottabyte.process.util import loggers
from src.cn.yottabyte.process.util import data_util
from src.cn.yottabyte.process.table import Table

logger = loggers.get_logger().getChild('DPPHandler')

class DPPHandler(CentralizedHandler):
    k = "3"
    QIANFAN_TOKEN_FILE = '/tmp/spl_dpp_access_token.json'
    QIANFAN_API_KEY = '填你的百度千帆应用apikey'
    QIANFAN_SECRET_KEY = '填你的百度千帆应用secretkey'
    DASHSCOPE_API_KEY = '填你的阿里云灵积apikey'

    def initialize(self, meta):
        self.access_token = None
        self.by = None
        # 获取自定义指令运行参数：
        # k 来代表每个聚类的采样数量
        # by_field 代表是输出每个聚类总结还是全局总结
        args_dict, args_list = data_util.get_args_from_meta(meta)
        if args_dict:
            self.k = args_dict.get("k", "3")
            self.by = args_dict.get("by_field")
        logger.info(f"call get info with {meta}")
        return {'type': 'centralized'}

    def get_access_token(self):
        # 检查文件中是否有有效的access_token
        if os.path.exists(self.QIANFAN_TOKEN_FILE):
            with open(self.QIANFAN_TOKEN_FILE, 'r') as f:
                token_data = json.load(f)
                if time.time() < token_data['expires_at']:
                    logger.info(time.time())
                    self.access_token = token_data['access_token']
                    return

        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.QIANFAN_API_KEY}&client_secret={self.QIANFAN_SECRET_KEY}"
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        logger.info('call get qianfan access token')
        response = requests.request("POST", url, headers=headers, data=payload)
        response_json = response.json()
        self.access_token = response_json.get("access_token")
        expires_in = response_json.get("expires_in", 3600)
        expires_at = time.time() + expires_in

        # 将新的access_token和过期时间写入文件
        with open(self.QIANFAN_TOKEN_FILE, 'w') as f:
            json.dump({'access_token': self.access_token, 'expires_at': expires_at}, f)


    def qianfan_summarize(self, content):
        self.get_access_token()
        url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed?access_token={self.access_token}"
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        })
        headers = {
            'Content-Type': 'application/json'
        }
        logger.info('call get qianfan ernie-speed llm')
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get('result','')

    def dashscope_summarize(self, content):
        url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        headers = {
            'Authorization': f"Bearer {self.DASHSCOPE_API_KEY}",
            'Content-Type': 'application/json'
        }
        data = {
            'model': 'qwen-plus',
            'input': {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': content
                    }
                ]
            },
            'parameters': {
                'result_format': 'text'
            }
        }
        logger.info('call get dashscope qwen-plus llm')
        response = requests.post(url, headers=headers, json=data)
        return response.json().get('output',{}).get('text','')

    def llm_summarize(self, content, modelname):
        if modelname == "qwen":
            return self.dashscope_summarize(content)
        else:
            return self.qianfan_summarize(content)

    def dpp_sample(self, S, k):
        # S: similarity matrix
        # k: number of items to sample
        n = S.shape[0]
        # Initialize empty set Y
        Y = set()
        for _ in range(k):
            best_i = -1
            best_p = -1
            for i in range(n):
                if i not in Y:
                    # Compute determinant of submatrix
                    det_Yi = np.linalg.det(S[np.ix_(list(Y) + [i], list(Y) + [i])])
                    # Compute probability of adding i to Y
                    p_add = det_Yi / (1 + det_Yi)
                    if p_add > best_p:
                        best_p = p_add
                        best_i = i
            # Add best item to Y
            Y.add(best_i)
        return list(Y)

    def process_cluster(self, cluster_rows, features):
        # 提取 tfidf 特征值，构建 numpy 数组
        feature_values = np.array([[float(row[feature]) for feature in features] for row in cluster_rows])
        # 使用 sklearn 中的方法构建相似性矩阵
        S = cosine_similarity(feature_values)
        # 应用 DPP 采样
        sampled_indices = self.dpp_sample(S, int(self.k))
        sampled_rows = [cluster_rows[i] for i in sampled_indices]
        # 发送采样日志，由大模型生成摘要
        content_parts = [
            "你是 IT 运维和网络安全专家，请总结下面这段日志内容，输出尽量简短、非结构化、保留关键信息："
        ] + [row['raw_message'] for row in sampled_rows]
        content = "\n".join(content_parts)
        summary = self.llm_summarize(content, 'ernie')
        return summary

    def execute(self, meta, table):
        # 检查是否存在 cluster 字段
        cluster_field_present = 'cluster' in table.fields
        if cluster_field_present:
            # 按聚类分组
            clusters = {}
            for row in table.rows:
                cluster_id = row['cluster']
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(row)
        else:
            clusters = {'0': table.rows}
        # 获取SPL输入的数据表中，有哪些 tfidf 特征向量字段
        features = [field for field in table.fields if 'tfidf' in field]

        finished = meta.get("finished", False)
        if finished:
            # 准备输出给SPL后续处理的数据表
            table = Table()
            table.fields = ['cluster', 'summary']
            # cluster分组内数据，并发应用DPP采样和 LLM 总结
            with ThreadPoolExecutor(max_workers=5) as executor:  # 可以调整线程数
                # 构建并发任务
                future_to_cluster = {
                    executor.submit(self.process_cluster, cluster_rows, features): cluster_id
                    for cluster_id, cluster_rows in clusters.items()
                }
                # 收集结果，追加到准备好的SPL数据表里
                for future in concurrent.futures.as_completed(future_to_cluster):
                    cluster_id = future_to_cluster[future]
                    try:
                        summary = future.result()
                        table.add_row({'cluster': cluster_id, 'summary': summary})
                    except Exception as exc:
                        logger.error(f'Cluster {cluster_id} generated an exception: {exc}')

            # 不要求分组输出，二次总结
            if not self.by:
                total_table = Table()
                total_table.fields = ['log_summary']
                total_content_parts = [
                    "你是 IT 运维和网络安全专家，下面是日志聚类后的关键信息摘要，请通盘考虑，输出中文总结和分析建议："
                ] + [row['summary'] for row in table.get_rows()]
                # 聚类总结内容已经是多行文本了，不能简单的用换行来合并 prompt，必须用明确分割符来指明每段文本
                total_content = "\n\n## 聚类摘要\n\n".join(total_content_parts)
                total_summary = self.llm_summarize(total_content, 'qwen')
                if total_summary is None:
                    logger.info("无法生成全局总结，请检查聚类总结内容。")
                total_table.add_row({'log_summary': total_summary})
                return meta, total_table
            else:
                return meta, table
        else:
            table = Table()
            return meta, table

if __name__ == "__main__":
    handler = DPPHandler()
    handler.run()
    handler.close()

