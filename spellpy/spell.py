import re
import os
import sys
import pickle
import signal
import csv
import pandas as pd
import hashlib
import math
import time
from collections import defaultdict
from datetime import datetime
import string
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


sys.setrecursionlimit(10000)

NON_ASCII_RE = re.compile(r'[^\x00-\x7F]+')
TOKEN_SPLIT_RE = re.compile(r'[\s=:,]')
DEFAULT_PROGRESS_INTERVAL = 50000


class LCSObject:
    """ Class object to store a log group with the same template
    """
    def __init__(self, logTemplate='', logIDL=None):
        self.logTemplate = logTemplate
        self.logIDL = [] if logIDL is None else list(logIDL)
        self.logIDSet = set(self.logIDL)
        self.occurrence_count = len(self.logIDL)


class Node:
    """ A node in prefix tree data structure
    """
    def __init__(self, token='', templateNo=0):
        self.logClust = None
        self.token = token
        self.templateNo = templateNo
        self.childD = dict()


class CustomUnpickler(pickle.Unpickler):
    """ CustomUnpickler is to prevent can't get attribute error when pickle load.
    """
    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class LogParser(pickle.Unpickler):
    """ LogParser class
    Attributes
    ----------
        path : the path of the input file
        logName : the file name of the input file
        savePath : the path of the output file
        tau : how much percentage of tokens matched to merge a log message
    """
    def __init__(
        self,
        indir='./',
        outdir='./result/',
        log_format=None,
        tau=0.5,
        keep_para=True,
        text_max_length=4096,
        logmain=None,
        date_filter='',
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
        max_lcs_comparisons_per_line=None,
        resume_state=False,
        slow_line_threshold=1.0,
        persist_state=False,
    ):
        self.path = indir
        self.logname = None
        self.logmain = logmain
        self.savePath = outdir
        self.tau = tau
        self.logformat = log_format
        self.keep_para = keep_para
        self.lastestLineId = 0
        self.text_max_length = text_max_length
        self.date_filter = date_filter
        self.progress_interval = max(0, progress_interval)
        self.max_lcs_comparisons_per_line = max_lcs_comparisons_per_line
        self.resume_state = resume_state
        self.slow_line_threshold = slow_line_threshold
        self.persist_state = persist_state
        self.parse_metrics = {}
        self.rootNode = Node()
        self.logCluL = []
        self._state_initialized = False
        self._token_to_clusters = defaultdict(set)
        self._wildcard_clusters = set()
        self._cluster_meta = {}
        self._cluster_order = {}
        self._next_cluster_order = 0
        self._cluster_index = {}
        self._next_cluster_index = 0

    def _normalize_cluster_history(self, cluster):
        logidl = getattr(cluster, 'logIDL', None)
        if logidl is None:
            cluster.logIDL = []
        elif isinstance(logidl, list):
            cluster.logIDL = logidl
        else:
            cluster.logIDL = list(logidl)

        logid_set = getattr(cluster, 'logIDSet', None)
        if logid_set is None:
            cluster.logIDSet = set(cluster.logIDL)
        else:
            cluster.logIDSet = set(logid_set)
            cluster.logIDSet.update(cluster.logIDL)
        cluster.occurrence_count = max(
            getattr(cluster, 'occurrence_count', 0),
            len(cluster.logIDL),
            len(cluster.logIDSet),
        )
        return cluster

    def _cluster_has_log_id(self, cluster, log_id):
        return log_id in getattr(cluster, 'logIDSet', set())

    def _record_cluster_log_id(self, cluster, log_id):
        if not hasattr(cluster, 'logIDSet') or cluster.logIDSet is None:
            cluster.logIDSet = set(cluster.logIDL)
        if log_id in cluster.logIDSet:
            return False
        cluster.logIDSet.add(log_id)
        cluster.logIDL.append(log_id)
        cluster.occurrence_count = max(getattr(cluster, 'occurrence_count', 0), len(cluster.logIDL))
        return True

    def _record_cluster_occurrence(self, cluster):
        cluster.occurrence_count = getattr(cluster, 'occurrence_count', len(cluster.logIDL)) + 1
        return cluster.occurrence_count

    def _event_row_for_cluster(self, cluster):
        template_str = ' '.join(cluster.logTemplate)
        eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
        return eid, template_str, getattr(cluster, 'occurrence_count', len(cluster.logIDL))

    def _write_template_summary(self, output_path, logCluL):
        with open(output_path, 'w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(['EventId', 'EventTemplate', 'Occurrences'])
            for cluster in logCluL:
                eid, template_str, occurrences = self._event_row_for_cluster(cluster)
                writer.writerow([eid, template_str, occurrences])

    def _finalize_structured_csv(self, source_path, output_path, row_cluster_indices, logCluL, append=False):
        first_chunk = not append or not os.path.exists(output_path) or os.path.getsize(output_path) == 0
        with open(source_path, 'r', newline='') as source, open(output_path, 'a' if append else 'w', newline='') as output:
            reader = csv.DictReader(source)
            fieldnames = reader.fieldnames or []
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            if first_chunk:
                writer.writeheader()
            for row, cluster_index in zip(reader, row_cluster_indices):
                cluster = logCluL[cluster_index]
                event_id, event_template, _ = self._event_row_for_cluster(cluster)
                row['EventId'] = event_id
                row['EventTemplate'] = event_template
                if self.keep_para:
                    row['ParameterList'] = self.get_parameter_list({'EventTemplate': event_template, 'Content': row['Content']})
                writer.writerow(row)

    def _template_stats(self, template):
        const_tokens = [token for token in template if token != '<*>']
        unique_const_tokens = set(const_tokens)
        return {
            'const_count': len(const_tokens),
            'token_set': unique_const_tokens,
            'template_len': len(template),
            'first_const_token': next((token for token in template if token != '<*>'), None),
        }

    def _register_cluster(self, cluster):
        stats = self._template_stats(cluster.logTemplate)
        self._cluster_meta[cluster] = stats
        if cluster not in self._cluster_order:
            self._cluster_order[cluster] = self._next_cluster_order
            self._next_cluster_order += 1
        if cluster not in self._cluster_index:
            self._cluster_index[cluster] = self._next_cluster_index
            self._next_cluster_index += 1

        if stats['const_count'] == 0:
            self._wildcard_clusters.add(cluster)
            return

        for token in stats['token_set']:
            self._token_to_clusters[token].add(cluster)

    def _unregister_cluster(self, cluster):
        stats = self._cluster_meta.pop(cluster, None)
        if stats is None:
            self._wildcard_clusters.discard(cluster)
            return

        if stats['const_count'] == 0:
            self._wildcard_clusters.discard(cluster)
            return

        for token in stats['token_set']:
            cluster_set = self._token_to_clusters.get(token)
            if cluster_set is None:
                continue
            cluster_set.discard(cluster)
            if not cluster_set:
                del self._token_to_clusters[token]

    def _rebuild_match_indexes(self, logCluL):
        self._token_to_clusters = defaultdict(set)
        self._wildcard_clusters = set()
        self._cluster_meta = {}
        self._cluster_index = {}
        self._next_cluster_index = 0
        for cluster in logCluL:
            self._normalize_cluster_history(cluster)
            self._register_cluster(cluster)

    def _tokenize_content(self, content):
        return [token for token in TOKEN_SPLIT_RE.split(content) if token]

    def _candidate_clusters(self, logCluL, seq, seq_token_set, for_lcs=False):
        if not logCluL:
            return []

        if not self._cluster_meta:
            return list(logCluL)

        if not seq_token_set:
            return [cluster for cluster in logCluL if cluster in self._wildcard_clusters]

        if for_lcs:
            unique_seq_tokens = list(dict.fromkeys(seq))
            unique_token_count = len(unique_seq_tokens)
            required_overlap = int(math.ceil(self.tau * len(seq)))
            if unique_token_count < required_overlap:
                return []

            pivot_count = max(1, unique_token_count - required_overlap + 1)
            token_freq = self._token_to_clusters
            pivot_tokens = sorted(
                unique_seq_tokens,
                key=lambda token: (len(token_freq.get(token, ())), token),
            )[:pivot_count]
        else:
            pivot_tokens = list(seq_token_set)

        candidate_set = set(self._wildcard_clusters)
        for token in pivot_tokens:
            candidate_set.update(self._token_to_clusters.get(token, ()))

        return sorted(candidate_set, key=lambda cluster: self._cluster_order.get(cluster, 0))

    def LCS(self, seq1, seq2):
        lengths = [[0 for j in range(len(seq2)+1)] for i in range(len(seq1)+1)]
        # row 0 and column 0 are initialized to 0 already
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                if seq1[i] == seq2[j]:
                    lengths[i+1][j+1] = lengths[i][j] + 1
                else:
                    lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])

        # read the substring out from the matrix
        result = []
        lenOfSeq1, lenOfSeq2 = len(seq1), len(seq2)
        while lenOfSeq1 != 0 and lenOfSeq2 != 0:
            if lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1-1][lenOfSeq2]:
                lenOfSeq1 -= 1
            elif lengths[lenOfSeq1][lenOfSeq2] == lengths[lenOfSeq1][lenOfSeq2-1]:
                lenOfSeq2 -= 1
            else:
                assert seq1[lenOfSeq1-1] == seq2[lenOfSeq2-1]
                result.insert(0, seq1[lenOfSeq1-1])
                lenOfSeq1 -= 1
                lenOfSeq2 -= 1
        return result

    def SimpleLoopMatch(self, logClustL, seq, seq_token_set=None):
        if seq_token_set is None:
            seq_token_set = set(seq)
        candidate_clusters = self._candidate_clusters(logClustL, seq, seq_token_set, for_lcs=False)
        if not candidate_clusters and self._cluster_meta:
            return None
        for logClust in candidate_clusters:
            if float(len(logClust.logTemplate)) < 0.5 * len(seq):
                continue
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            if all(token in seq_token_set or token == '<*>' for token in logClust.logTemplate):
                return logClust
        return None

    def PrefixTreeMatch(self, parentn, seq, idx):
        retLogClust = None
        length = len(seq)
        for i in range(idx, length):
            if seq[i] in parentn.childD:
                childn = parentn.childD[seq[i]]
                if (childn.logClust is not None):
                    constLM = [w for w in childn.logClust.logTemplate if w != '<*>']
                    if float(len(constLM)) >= self.tau * length:
                        return childn.logClust
                else:
                    return self.PrefixTreeMatch(childn, seq, i + 1)

        return retLogClust

    def LCSMatch(self, LCSMap, seq, seq_token_set=None, metrics=None):
        retLCSObject = None

        maxLen = -1
        maxLCSObject = None
        if seq_token_set is None:
            seq_token_set = set(seq)
        size_seq = len(seq)
        candidate_clusters = self._candidate_clusters(LCSMap, seq, seq_token_set, for_lcs=True)
        comparisons = 0
        skipped_by_guardrail = 0
        required_overlap = int(math.ceil(self.tau * size_seq))
        seq_first_token = seq[0] if seq else None

        def candidate_sort_key(cluster):
            stats = self._cluster_meta.get(cluster)
            if stats is None:
                stats = self._template_stats(cluster.logTemplate)
            first_token_rank = 0 if seq_first_token is not None and stats['first_const_token'] == seq_first_token else 1
            return (first_token_rank, -stats['const_count'], stats['template_len'])

        ordered_candidates = sorted(candidate_clusters, key=candidate_sort_key)

        for LCSObject in ordered_candidates:
            if self.max_lcs_comparisons_per_line is not None and comparisons >= self.max_lcs_comparisons_per_line:
                skipped_by_guardrail = len(candidate_clusters) - comparisons
                logger.warning(
                    'LCS guardrail triggered after %s comparisons for a line; skipping %s remaining candidates.',
                    comparisons,
                    skipped_by_guardrail,
                )
                break

            set_template = self._cluster_meta.get(LCSObject, {}).get('token_set')
            if set_template is None:
                set_template = {token for token in LCSObject.logTemplate if token != '<*>'}
            const_count = len(set_template)
            template_len = len(LCSObject.logTemplate)
            if len(seq_token_set & set_template) < required_overlap:
                continue
            if maxLen >= 0:
                if const_count < maxLen:
                    continue
                if const_count == maxLen and maxLCSObject is not None and template_len >= len(maxLCSObject.logTemplate):
                    continue
            comparisons += 1
            if metrics is not None:
                metrics['total_lcs_comparisons'] += 1
            lcs = self.LCS(seq, LCSObject.logTemplate)
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(LCSObject.logTemplate) < len(maxLCSObject.logTemplate)):
                maxLen = len(lcs)
                maxLCSObject = LCSObject

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLCSObject = maxLCSObject

        if metrics is not None and skipped_by_guardrail:
            metrics['guardrail_skips'] += skipped_by_guardrail

        return retLCSObject

    def getTemplate(self, lcs, seq):
        retVal = []
        if not lcs:
            return retVal

        lcs = lcs[::-1]
        i = 0
        for token in seq:
            i += 1
            if token == lcs[-1]:
                retVal.append(token)
                lcs.pop()
            else:
                retVal.append('<*>')
            if not lcs:
                break
        if i < len(seq):
            retVal.append('<*>')
        return retVal

    def addSeqToPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '<*>']

        for i in range(len(seq)):
            tokenInSeq = seq[i]
            # Match
            if tokenInSeq in parentn.childD:
                parentn.childD[tokenInSeq].templateNo += 1
            # Do not Match
            else:
                parentn.childD[tokenInSeq] = Node(token=tokenInSeq, templateNo=1)
            parentn = parentn.childD[tokenInSeq]

        if parentn.logClust is None:
            parentn.logClust = newCluster

    def removeSeqFromPrefixTree(self, rootn, newCluster):
        parentn = rootn
        seq = newCluster.logTemplate
        seq = [w for w in seq if w != '<*>']

        for tokenInSeq in seq:
            if tokenInSeq in parentn.childD:
                matchedNode = parentn.childD[tokenInSeq]
                if matchedNode.templateNo == 1:
                    del parentn.childD[tokenInSeq]
                    break
                else:
                    matchedNode.templateNo -= 1
                    parentn = matchedNode

    def parse(self, logname):
        starttime = datetime.now()
        logger.info('Parsing file: ' + os.path.join(self.path, logname))
        self.logname = logname
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        rootNodePath = os.path.join(self.savePath, 'rootNode.pkl')
        logCluLPath = os.path.join(self.savePath, 'logCluL.pkl')
        stateMetaPath = os.path.join(self.savePath, 'state_meta.pkl')

        if not self._state_initialized:
            if self.resume_state and os.path.exists(rootNodePath) and os.path.exists(logCluLPath):
                with open(rootNodePath, 'rb') as f:
                    self.rootNode = CustomUnpickler(f).load()
                with open(logCluLPath, 'rb') as f:
                    self.logCluL = CustomUnpickler(f).load()
                for logclust in self.logCluL:
                    self._normalize_cluster_history(logclust)
                if os.path.exists(stateMetaPath):
                    with open(stateMetaPath, 'rb') as f:
                        state_meta = pickle.load(f)
                    self.lastestLineId = int(state_meta.get('lastestLineId', 0))
                else:
                    self.lastestLineId = max(
                        (max(logclust.logIDSet) for logclust in self.logCluL if getattr(logclust, 'logIDSet', None)),
                        default=0,
                    )
                logger.info(f'Load objects done, lastestLineId: {self.lastestLineId}')
            else:
                self.rootNode = Node()
                self.logCluL = []
                self.lastestLineId = 0
            self._state_initialized = True

        rootNode = self.rootNode
        logCluL = self.logCluL

        if not self.resume_state and not self.logCluL:
            self.lastestLineId = 0

        self._rebuild_match_indexes(logCluL)

        log_file = os.path.join(self.path, self.logname)
        headers, regex = self.generate_logformat_regex(self.logformat)
        content_idx = headers.index('Content')
        total_line = self._count_lines(log_file)

        count = 0
        metrics = {
            'raw_lines_seen': 0,
            'input_lines_processed': 0,
            'templates_created': 0,
            'total_lcs_comparisons': 0,
            'candidate_templates_sum': 0,
            'candidate_templates_max': 0,
            'guardrail_skips': 0,
            'duplicate_membership_checks': 0,
            'duplicate_membership_hits': 0,
            'line_elapsed_total_seconds': 0.0,
            'line_elapsed_max_seconds': 0.0,
            'slow_lines': 0,
            'max_cluster_history_size': max(
                (getattr(cluster, 'occurrence_count', len(cluster.logIDL)) for cluster in logCluL),
                default=0,
            ),
            'max_cluster_history_set_size': max((len(cluster.logIDSet) for cluster in logCluL), default=0),
        }

        output_path = os.path.join(self.savePath, self.logname + '_structured.csv')
        temp_output_path = output_path + '.tmp'
        main_output_path = None
        temp_main_output_path = None
        if self.logmain:
            main_output_path = os.path.join(self.savePath, self.logmain + '_main_structured.csv')
            temp_main_output_path = main_output_path + '.tmp'

        output_headers = ['LineId'] + headers + ['EventId', 'EventTemplate']
        if self.keep_para:
            output_headers.append('ParameterList')

        row_cluster_indices = []
        with open(temp_output_path, 'w', newline='') as structured_file:
            structured_writer = csv.writer(structured_file)
            structured_writer.writerow(output_headers)

            main_writer = None
            main_file = None
            if temp_main_output_path:
                main_file = open(temp_main_output_path, 'w', newline='')
                main_writer = csv.writer(main_file)
                main_writer.writerow(output_headers)

            try:
                with open(log_file, 'r') as fin:
                    for raw_line in fin:
                        metrics['raw_lines_seen'] += 1
                        if len(raw_line) > self.text_max_length:
                            logger.error('Length of log string is too long')
                            logger.error(raw_line)
                            continue
                        if self.date_filter not in raw_line:
                            continue

                        signal.signal(signal.SIGALRM, self._log_to_dataframe_handler)
                        signal.alarm(1)
                        try:
                            line = NON_ASCII_RE.sub('<NASCII>', raw_line)
                            match = regex.search(line.strip())
                            if match is None:
                                continue
                            message = [match.group(header) for header in headers]
                        except Exception as e:
                            _ = e
                            continue
                        finally:
                            signal.alarm(0)

                        logID = self.lastestLineId + count + 1
                        line_start = time.perf_counter()
                        logmessageL = self._tokenize_content(message[content_idx])
                        constLogMessL = [w for w in logmessageL if w != '<*>']
                        seq_token_set = set(constLogMessL)
                        candidate_clusters = self._candidate_clusters(logCluL, constLogMessL, seq_token_set, for_lcs=False)
                        metrics['candidate_templates_sum'] += len(candidate_clusters)
                        metrics['candidate_templates_max'] = max(metrics['candidate_templates_max'], len(candidate_clusters))

                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                'Matching line %s: clusters=%s candidates=%s',
                                logID,
                                len(logCluL),
                                len(candidate_clusters),
                            )

                        matchCluster = self.PrefixTreeMatch(rootNode, constLogMessL, 0)

                        if matchCluster is None:
                            matchCluster = self.SimpleLoopMatch(candidate_clusters, constLogMessL, seq_token_set=seq_token_set)

                            if matchCluster is None:
                                matchCluster = self.LCSMatch(logCluL, logmessageL, seq_token_set=seq_token_set, metrics=metrics)

                                if matchCluster is None:
                                    matchCluster = LCSObject(logTemplate=logmessageL, logIDL=[])
                                    matchCluster.occurrence_count = 0
                                    logCluL.append(matchCluster)
                                    self._register_cluster(matchCluster)
                                    metrics['templates_created'] += 1
                                    self.addSeqToPrefixTree(rootNode, matchCluster)
                                else:
                                    newTemplate = self.getTemplate(self.LCS(logmessageL, matchCluster.logTemplate),
                                                                   matchCluster.logTemplate)
                                    if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                                        self.removeSeqFromPrefixTree(rootNode, matchCluster)
                                        self._unregister_cluster(matchCluster)
                                        matchCluster.logTemplate = newTemplate
                                        self._register_cluster(matchCluster)
                                        self.addSeqToPrefixTree(rootNode, matchCluster)

                        if matchCluster is not None:
                            self._record_cluster_occurrence(matchCluster)

                        cluster_index = self._cluster_index.get(matchCluster)
                        if cluster_index is None:
                            try:
                                cluster_index = logCluL.index(matchCluster)
                            except ValueError:
                                logCluL.append(matchCluster)
                                cluster_index = len(logCluL) - 1
                            self._cluster_index[matchCluster] = cluster_index
                            self._next_cluster_index = max(self._next_cluster_index, cluster_index + 1)
                        row_cluster_indices.append(cluster_index)
                        placeholder_row = [logID] + message + ['0', '']
                        if self.keep_para:
                            placeholder_row.append([])
                        structured_writer.writerow(placeholder_row)
                        if main_writer is not None:
                            main_writer.writerow(placeholder_row)

                        line_elapsed = time.perf_counter() - line_start
                        metrics['line_elapsed_total_seconds'] += line_elapsed
                        metrics['line_elapsed_max_seconds'] = max(metrics['line_elapsed_max_seconds'], line_elapsed)
                        if self.slow_line_threshold is not None and line_elapsed >= self.slow_line_threshold:
                            metrics['slow_lines'] += 1
                            logger.warning(
                                'Slow line %s took %.3fs; clusters=%s candidates=%s lcs=%s duplicate_checks=%s history_size=%s',
                                logID,
                                line_elapsed,
                                len(logCluL),
                                len(candidate_clusters),
                                metrics['total_lcs_comparisons'],
                                metrics['duplicate_membership_checks'],
                                metrics['max_cluster_history_size'],
                            )

                        count += 1
                        metrics['input_lines_processed'] += 1
                        metrics['max_cluster_history_size'] = max(
                            metrics['max_cluster_history_size'],
                            getattr(matchCluster, 'occurrence_count', 0),
                        )
                        if self.progress_interval and (count % self.progress_interval == 0 or count == total_line):
                            structured_file.flush()
                            if main_file is not None:
                                main_file.flush()
                            mean_candidates = (
                                metrics['candidate_templates_sum'] / metrics['input_lines_processed']
                                if metrics['input_lines_processed'] else 0.0
                            )
                            percentage = count * 100.0 / total_line if total_line else 100.0
                            logger.info(
                                'Processed %.1f%% of log lines. lines=%s templates=%s lcs=%s candidates_mean=%.2f candidates_max=%s guardrail_skips=%s duplicate_checks=%s',
                                percentage,
                                metrics['input_lines_processed'],
                                len(logCluL),
                                metrics['total_lcs_comparisons'],
                                mean_candidates,
                                metrics['candidate_templates_max'],
                                metrics['guardrail_skips'],
                                metrics['duplicate_membership_checks'],
                            )
            finally:
                if main_file is not None:
                    main_file.close()

        self._finalize_structured_csv(temp_output_path, output_path, row_cluster_indices, logCluL, append=False)
        try:
            os.remove(temp_output_path)
        except OSError:
            pass

        if temp_main_output_path is not None:
            self._finalize_structured_csv(temp_main_output_path, main_output_path, row_cluster_indices, logCluL, append=True)
            try:
                os.remove(temp_main_output_path)
            except OSError:
                pass

        self.lastestLineId += metrics['input_lines_processed']

        templates_path = os.path.join(self.savePath, self.logname + '_templates.csv')
        self._write_template_summary(templates_path, logCluL)
        if self.logmain:
            main_templates_path = os.path.join(self.savePath, self.logmain + '_main_templates.csv')
            self._write_template_summary(main_templates_path, logCluL)

        if self.persist_state:
            logger.info(f'rootNodePath: {rootNodePath}')
            with open(rootNodePath, 'wb') as output:
                pickle.dump(rootNode, output, pickle.HIGHEST_PROTOCOL)
            logger.info(f'logCluLPath: {logCluLPath}')
            with open(logCluLPath, 'wb') as output:
                pickle.dump(logCluL, output, pickle.HIGHEST_PROTOCOL)
            with open(stateMetaPath, 'wb') as output:
                pickle.dump({'lastestLineId': self.lastestLineId}, output, pickle.HIGHEST_PROTOCOL)
            logger.info('Store objects done.')

        elapsed = (datetime.now() - starttime).total_seconds()
        mean_candidates = (
            metrics['candidate_templates_sum'] / metrics['input_lines_processed']
            if metrics['input_lines_processed'] else 0.0
        )
        self.parse_metrics = {
            **metrics,
            'candidate_templates_mean': mean_candidates,
            'elapsed_seconds': elapsed,
            'templates_total': len(logCluL),
            'cluster_count': len(logCluL),
        }
        logger.info(
            'Parsing done. [Time taken: %s] metrics=%s',
            datetime.now() - starttime,
            self.parse_metrics,
        )

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        total_line = self._count_lines(log_file)

        with open(log_file, 'r') as fin:
            for line in fin:
                if len(line) > self.text_max_length:
                    logger.error('Length of log string is too long')
                    logger.error(line)
                    continue
                if self.date_filter not in line:
                    # logging.warning(f'{self.date_filter} is not in {line}')
                    continue
                signal.signal(signal.SIGALRM, self._log_to_dataframe_handler)
                signal.alarm(1)
                line = NON_ASCII_RE.sub('<NASCII>', line)
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                    if linecount % DEFAULT_PROGRESS_INTERVAL == 0 or linecount == total_line:
                        logger.info('Loaded {0:.1f}% of log lines.'.format(linecount*100/total_line))
                except Exception as e:
                    _ = e
                    pass
                signal.alarm(0)
        df_log = pd.DataFrame(log_messages, columns=headers)
        df_log.insert(0, 'LineId', None)
        df_log['LineId'] = [i + 1 for i in range(linecount)]
        return df_log

    def _count_lines(self, log_file):
        total_line = 0
        with open(log_file, 'r') as fin:
            for _ in fin:
                total_line += 1
        return total_line

    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(r'\\ +', r' ', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += f'(?P<{header}>.*?)'
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def get_parameter_list(self, row):
        event_template = str(row["EventTemplate"])
        template_regex = re.sub(r"\s<.{1,5}>\s", "<*>", event_template)
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r'([^A-Za-z0-9])', r'\\\1', template_regex)
        template_regex = re.sub(r'\\ +', r'[^A-Za-z0-9]+', template_regex)
        template_regex = "^" + template_regex.replace(r"\<\*\>", "(.*?)") + "$"

        signal.signal(signal.SIGALRM, self._parameter_handler)
        signal.alarm(1)
        try:
            parameter_list = self._get_parameter_list(row, template_regex)
        except Exception as e:
            logger.error(e)
            parameter_list = ["TIMEOUT"]
        signal.alarm(0)
        return parameter_list

    def _get_parameter_list(self, row, template_regex):
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        parameter_list = list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]
        parameter_list = [para.strip(string.punctuation).strip(' ') for para in parameter_list]
        return parameter_list

    def _parameter_handler(self, signum, frame):
        logger.error("_get_parameter_list function is hangs!")
        raise Exception("TIME OUT!")

    def _log_to_dataframe_handler(self, signum, frame):
        logger.error('log_to_dataframe function is hangs')
        raise Exception("TIME OUT!")
