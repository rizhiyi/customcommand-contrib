#!python3
import os
import sys
from pysenti import ModelClassifier

executehome = "/opt/rizhiyi/parcels/splserver"
lib_path = os.path.join(executehome, 'bin', 'custom_commands', 'lib')
sys.path.insert(0, lib_path)

from src.cn.yottabyte.process.streaming_handler import StreamingHandler
from src.cn.yottabyte.process.util import loggers
from src.cn.yottabyte.process.util import data_util
from src.cn.yottabyte.process.table import Table

logger = loggers.get_logger().getChild('SentimentHandler')

class SentimentHandler(StreamingHandler):

    def initialize(self, meta):
        args_dict, args_list = data_util.get_args_from_meta(meta)
        if args_dict:
            self.input = args_dict.get("input", "input")
        self.model = ModelClassifier()
        logger.info(f"call get info with {meta}")
        return {'type': 'streaming'}
    
    def execute(self, meta, table):
        out_csv = self.executeRow(table)
        return meta, out_csv

    def executeRow(self, table):
        table.add_field('label')
        table.add_field('score')
        for row in table.rows:
            text = row[self.input]
            result = self.model.classify(text)
            if result['positive_prob'] > result['negative_prob']:
                row['label'] = 'positive'
                row['score'] = result['positive_prob']
            else:
                row['label'] = 'negative'
                row['score'] = result['negative_prob']
        return table

if __name__ == "__main__":
    handler = SentimentHandler()
    handler.run()
    handler.close()

