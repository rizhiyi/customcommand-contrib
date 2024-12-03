#!python3
import os
import sys
from scipy.stats import ks_2samp

executehome = "/opt/rizhiyi/parcels/splserver"
lib_path = os.path.join(executehome, 'bin', 'custom_commands', 'lib')
sys.path.insert(0, lib_path)

from src.cn.yottabyte.process.centralized_handler import CentralizedHandler
from src.cn.yottabyte.process.util import loggers
from src.cn.yottabyte.process.util import data_util
from src.cn.yottabyte.process.table import Table

logger = loggers.get_logger().getChild('KS2SAMPHandler')

class KS2SAMPHandler(CentralizedHandler):

    def initialize(self, meta):
        # 获取自定义指令运行参数：
        args_dict, args_list = data_util.get_args_from_meta(meta)
        if args_list:
            self.data1 = args_list[0]
            self.data2 = args_list[1]
        logger.info(f"call get info with {meta}")
        return {'type': 'centralized'}

    def execute(self, meta, table):
        data1_values = []
        data2_values = []
        for row in table.rows:
            data1_values.append(row[self.data1])
            data2_values.append(row[self.data2])
        statistic, p_value = self.ks_2samp_test(data1_values, data2_values)
        if p_value < 0.05:
            decision = "Reject the null hypothesis that the independent samples are drawn from the same continuous distribution."
        else:
            decision = "Fail to reject the null hypothesis that the independent samples are drawn from the same continuous distribution."
        logger.info(f"KS test p-value: {p_value}")

        finished = meta.get("finished", False)
        if finished:
            # 准备输出给SPL后续处理的数据表
            table = Table()
            table.fields = ['statistic', 'p-value', 'Test decision(alpha=0.05)']
            table.add_row({'statistic': statistic, 'p-value': p_value, 'Test decision(alpha=0.05)': decision})
            return meta, table
        else:
            table = Table()
            return meta, table

    def ks_2samp_test(self, array1, array2):
        """
        Perform a two-sample Kolmogorov-Smirnov test on two arrays.
        
        :param array1: First array of data
        :param array2: Second array of data
        :return: p-value of the test
        """
        statistic, p_value = ks_2samp(array1, array2)
        logger.info(f"KS test statistic: {statistic}, p-value: {p_value}")
        return statistic, p_value

if __name__ == "__main__":
    handler = KS2SAMPHandler()
    handler.run()
    handler.close()

