import unittest
import re
import os
import tempfile
import pandas as pd
from unittest.mock import patch
from pandas.testing import assert_frame_equal
from spellpy.spell import LogParser, LCSObject, Node

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FORMAT = '<Date> <Time> <Pid> <Level> <Component>: <Content>'

mock = {
    'LineId': [1, 2, 3],
    'Date': ['081109', '081109', '081109'],
    'Time': ['203518', '203518', '203519'],
    'Pid': ['143', '35', '143'],
    'Level': ['INFO', 'INFO', 'INFO'],
    'Component': ['dfs.DataNode$DataXceiver', 'dfs.FSNamesystem', 'dfs.DataNode$DataXceiver'],
    'Content': [
        'Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010',
        'BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906',
        'Receiving block blk_-1608999687919862906 src: /10.250.10.6:40524 dest: /10.250.10.6:50010'
    ],
}
DF_MOCK = pd.DataFrame(mock)


class TestLogParser(unittest.TestCase):
    def setUp(self):
        self.parser = LogParser()

    def test_generate_logformat_regex(self):
        expected_header = ['Date', 'Time', 'Pid', 'Level', 'Component', 'Content']
        expected_regex = re.compile(
            '^(?P<Date>.*?) (?P<Time>.*?) (?P<Pid>.*?) (?P<Level>.*?) (?P<Component>.*?): (?P<Content>.*?)$'
        )

        header, regex = self.parser.generate_logformat_regex(LOG_FORMAT)
        self.assertListEqual(header, expected_header)
        self.assertCountEqual(header, expected_header)
        self.assertEqual(regex, expected_regex)

    def test_log_to_dataframe(self):
        test_data_path = os.path.join(THIS_DIR, 'test_data.log')
        header, regex = self.parser.generate_logformat_regex(LOG_FORMAT)
        df_log = self.parser.log_to_dataframe(
            test_data_path, regex, header, LOG_FORMAT
        )
        assert_frame_equal(df_log, DF_MOCK)

    def test_parse_streaming_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.parser = LogParser(
                indir=THIS_DIR,
                outdir=tmpdir,
                log_format=LOG_FORMAT,
                keep_para=False,
            )
            self.parser.parse('test_data.log')

            structured_path = os.path.join(self.parser.savePath, 'test_data.log_structured.csv')
            df_structured = pd.read_csv(structured_path)

            self.assertEqual(len(df_structured), 3)
            self.assertEqual(df_structured['EventTemplate'].nunique(), 2)
            self.assertListEqual(df_structured['LineId'].tolist(), [1, 2, 3])
            self.assertEqual(self.parser.parse_metrics['input_lines_processed'], 3)
            self.assertEqual(self.parser.parse_metrics['templates_created'], 2)
            self.assertGreaterEqual(self.parser.parse_metrics['candidate_templates_mean'], 0.0)
            self.assertIn('duplicate_membership_checks', self.parser.parse_metrics)
            self.assertIn('line_elapsed_max_seconds', self.parser.parse_metrics)

    def test_addSeqToPrefixTree(self):
        logmessageL = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.19.102', '54106', 'dest', '/10.250.19.102', '50010']
        logID = 0

        rootNode = Node()
        newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])

        self.parser.addSeqToPrefixTree(rootNode, newCluster)
        res = helper(rootNode)
        self.assertEqual(res, logmessageL)

    def test_LCS(self):
        seq1 = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.10.6', '40524', 'dest', '/10.250.10.6', '50010']
        seq2 = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.19.102', '54106', 'dest', '/10.250.19.102', '50010']
        expected_lcs = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', 'dest', '50010']

        lcs = self.parser.LCS(seq1, seq2)
        self.assertListEqual(lcs, expected_lcs)

    def test_LCSMatch(self):
        seq1 = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.10.6', '40524', 'dest', '/10.250.10.6', '50010']
        seq2 = ['Just', 'A', 'Test']
        logmessageL = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.19.102', '54106', 'dest', '/10.250.19.102', '50010']
        logID = 0
        newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])

        retLogClust = self.parser.LCSMatch([newCluster], seq1)
        self.assertListEqual(retLogClust.logTemplate, newCluster.logTemplate)

        ret = self.parser.LCSMatch([newCluster], seq2)
        self.assertEqual(ret, None)

    def test_getTemplate(self):
        lcs = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', 'dest', '50010']
        seq = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '/10.250.19.102', '54106', 'dest', '/10.250.19.102', '50010']
        expected_template = ['Receiving', 'block', 'blk_-1608999687919862906', 'src', '<*>', '<*>', 'dest', '<*>', '50010']

        new_template = self.parser.getTemplate(lcs, seq)
        self.assertListEqual(new_template, expected_template)

    def test_record_cluster_log_id_uses_constant_time_membership(self):
        class RaisingList(list):
            def __contains__(self, item):
                raise AssertionError('list membership should not be used')

        cluster = LCSObject(logTemplate=['alpha', 'beta'], logIDL=[1, 2])
        cluster.logIDL = RaisingList(cluster.logIDL)
        cluster.logIDSet = {1, 2}

        self.assertFalse(self.parser._record_cluster_log_id(cluster, 2))
        self.assertTrue(self.parser._record_cluster_log_id(cluster, 3))
        self.assertListEqual(list(cluster.logIDL), [1, 2, 3])
        self.assertSetEqual(cluster.logIDSet, {1, 2, 3})

    def test_lcs_candidate_filtering_reduces_calls_without_changing_match(self):
        parser = LogParser(tau=0.5)
        seq = ['alpha', 'beta', 'shared1', 'shared2', 'shared3', 'shared4']
        best = LCSObject(logTemplate=seq.copy(), logIDL=[1])
        distractors = [
            LCSObject(
                logTemplate=['alpha', 'beta', 'shared1', 'shared2', '<*>', 'shared3', 'shared4', '<*>'],
                logIDL=[i + 2],
            )
            for i in range(24)
        ]
        logCluL = [best] + distractors
        parser._rebuild_match_indexes(logCluL)

        call_count = {'count': 0}
        original_lcs = parser.LCS

        def counting_lcs(seq1, seq2):
            call_count['count'] += 1
            return original_lcs(seq1, seq2)

        with patch.object(parser, 'LCS', side_effect=counting_lcs):
            match = parser.LCSMatch(logCluL, seq, seq_token_set=set(seq))

        self.assertIs(match, best)
        self.assertLess(call_count['count'], len(logCluL))
        self.assertEqual(call_count['count'], 1)

    def test_lcs_guardrail_limits_comparisons_and_records_skips(self):
        parser = LogParser(tau=0.5, max_lcs_comparisons_per_line=1)
        seq = ['alpha', 'beta', 'shared1', 'shared2', 'shared3', 'shared4']
        best = LCSObject(logTemplate=seq.copy(), logIDL=[1])
        distractors = [
            LCSObject(
                logTemplate=['alpha', 'beta', 'shared1', 'shared2', '<*>', 'shared3', 'shared4', '<*>'],
                logIDL=[i + 2],
            )
            for i in range(10)
        ]
        logCluL = [best] + distractors
        parser._rebuild_match_indexes(logCluL)

        metrics = {'total_lcs_comparisons': 0, 'guardrail_skips': 0}
        call_count = {'count': 0}
        original_lcs = parser.LCS

        def counting_lcs(seq1, seq2):
            call_count['count'] += 1
            return original_lcs(seq1, seq2)

        with patch.object(parser, 'LCS', side_effect=counting_lcs):
            match = parser.LCSMatch(logCluL, seq, seq_token_set=set(seq), metrics=metrics)

        self.assertIs(match, best)
        self.assertEqual(call_count['count'], 1)
        self.assertEqual(metrics['total_lcs_comparisons'], 1)
        self.assertGreater(metrics['guardrail_skips'], 0)

    def test_parse_does_not_reuse_disk_state_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = LogParser(
                indir=THIS_DIR,
                outdir=tmpdir,
                log_format=LOG_FORMAT,
                keep_para=False,
            )
            first.parse('test_data.log')
            first_df = pd.read_csv(os.path.join(tmpdir, 'test_data.log_structured.csv'))
            self.assertListEqual(first_df['LineId'].tolist(), [1, 2, 3])

            second = LogParser(
                indir=THIS_DIR,
                outdir=tmpdir,
                log_format=LOG_FORMAT,
                keep_para=False,
            )
            second.parse('test_data.log')
            second_df = pd.read_csv(os.path.join(tmpdir, 'test_data.log_structured.csv'))
            self.assertListEqual(second_df['LineId'].tolist(), [1, 2, 3])

    def test_parse_can_resume_state_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = LogParser(
                indir=THIS_DIR,
                outdir=tmpdir,
                log_format=LOG_FORMAT,
                keep_para=False,
            )
            first.parse('test_data.log')

            resumed = LogParser(
                indir=THIS_DIR,
                outdir=tmpdir,
                log_format=LOG_FORMAT,
                keep_para=False,
                resume_state=True,
            )
            resumed.parse('test_data.log')
            resumed_df = pd.read_csv(os.path.join(tmpdir, 'test_data.log_structured.csv'))
            self.assertListEqual(resumed_df['LineId'].tolist(), [4, 5, 6])


def helper(rootNode):
    if rootNode.childD == dict():
        return []

    res = []
    for k in rootNode.childD.keys():
        res.append(k)
        res += helper(rootNode.childD[k])
    return res


if __name__ == '__main__':
    unittest.main()
