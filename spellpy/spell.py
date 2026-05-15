import re
import os
import sys
import pickle
import signal
import csv
import pandas as pd
import hashlib
from datetime import datetime
import string
import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s]: %(message)s')


sys.setrecursionlimit(10000)


class LCSObject:
    """ Class object to store a log group with the same template
    """
    def __init__(self, logTemplate='', logIDL=[]):
        self.logTemplate = logTemplate
        self.logIDL = logIDL


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
    def __init__(self, indir='./', outdir='./result/', log_format=None, tau=0.5, keep_para=True, text_max_length=4096, logmain=None, date_filter=''):
        self.path = indir
        self.logname = None
        self.logmain = logmain
        self.savePath = outdir
        self.tau = tau
        self.logformat = log_format
        self.df_log = None
        self.keep_para = keep_para
        self.lastestLineId = 0
        self.text_max_length = text_max_length
        self.date_filter = date_filter

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

    def SimpleLoopMatch(self, logClustL, seq):
        for logClust in logClustL:
            if float(len(logClust.logTemplate)) < 0.5 * len(seq):
                continue
            # Check the template is a subsequence of seq (we use set checking as a proxy here for speedup since
            # incorrect-ordering bad cases rarely occur in logs)
            token_set = set(seq)
            if all(token in token_set or token == '<*>' for token in logClust.logTemplate):
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

    def LCSMatch(self, LCSMap, seq):
        retLCSObject = None

        maxLen = -1
        maxLCSObject = None
        set_seq = set(seq)
        size_seq = len(seq)
        for LCSObject in LCSMap:
            set_template = set(LCSObject.logTemplate)
            if len(set_seq & set_template) < 0.5 * size_seq:
                continue
            lcs = self.LCS(seq, LCSObject.logTemplate)
            if len(lcs) > maxLen or (len(lcs) == maxLen and len(LCSObject.logTemplate) < len(maxLCSObject.logTemplate)):
                maxLen = len(lcs)
                maxLCSObject = LCSObject

        # LCS should be large then tau * len(itself)
        if float(maxLen) >= self.tau * size_seq:
            retLCSObject = maxLCSObject

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
        logging.info('Parsing file: ' + os.path.join(self.path, logname))
        self.logname = logname
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)

        rootNodePath = os.path.join(self.savePath, 'rootNode.pkl')
        logCluLPath = os.path.join(self.savePath, 'logCluL.pkl')

        if os.path.exists(rootNodePath) and os.path.exists(logCluLPath):
            with open(rootNodePath, 'rb') as f:
                rootNode = CustomUnpickler(f).load()
            with open(logCluLPath, 'rb') as f:
                logCluL = CustomUnpickler(f).load()
            self.lastestLineId = 0
            for logclust in logCluL:
                if max(logclust.logIDL) > self.lastestLineId:
                    self.lastestLineId = max(logclust.logIDL)
            logging.info(f'Load objects done, lastestLineId: {self.lastestLineId}')
        else:
            rootNode = Node()
            logCluL = []
            self.lastestLineId = 0

        log_file = os.path.join(self.path, self.logname)
        headers, regex = self.generate_logformat_regex(self.logformat)
        content_idx = headers.index('Content')
        cache_path = os.path.join(self.savePath, f'.{self.logname}.raw.csv')
        total_line = self._count_lines(log_file)

        count = 0
        with open(cache_path, 'w', newline='') as cache_file:
            writer = csv.writer(cache_file)
            writer.writerow(['LineId'] + headers)

            with open(log_file, 'r') as fin:
                for raw_line in fin:
                    if len(raw_line) > self.text_max_length:
                        logging.error('Length of log string is too long')
                        logging.error(raw_line)
                        continue
                    if self.date_filter not in raw_line:
                        continue

                    signal.signal(signal.SIGALRM, self._log_to_dataframe_handler)
                    signal.alarm(1)
                    try:
                        line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', raw_line)
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
                    writer.writerow([logID] + message)
                    logmessageL = list(filter(lambda x: x != '', re.split(r'[\s=:,]', message[content_idx])))
                    constLogMessL = [w for w in logmessageL if w != '<*>']

                    # Find an existing matched log cluster
                    matchCluster = self.PrefixTreeMatch(rootNode, constLogMessL, 0)

                    if matchCluster is None:
                        matchCluster = self.SimpleLoopMatch(logCluL, constLogMessL)

                        if matchCluster is None:
                            matchCluster = self.LCSMatch(logCluL, logmessageL)

                            # Match no existing log cluster
                            if matchCluster is None:
                                newCluster = LCSObject(logTemplate=logmessageL, logIDL=[logID])
                                logCluL.append(newCluster)
                                self.addSeqToPrefixTree(rootNode, newCluster)
                            # Add the new log message to the existing cluster
                            else:
                                newTemplate = self.getTemplate(self.LCS(logmessageL, matchCluster.logTemplate),
                                                               matchCluster.logTemplate)
                                if ' '.join(newTemplate) != ' '.join(matchCluster.logTemplate):
                                    self.removeSeqFromPrefixTree(rootNode, matchCluster)
                                    matchCluster.logTemplate = newTemplate
                                    self.addSeqToPrefixTree(rootNode, matchCluster)
                    if matchCluster and logID not in matchCluster.logIDL:
                        matchCluster.logIDL.append(logID)

                    count += 1
                    if count % 10000 == 0 or count == total_line:
                        logging.info('Processed {0:.1f}% of log lines.'.format(count * 100.0 / total_line))

        self._write_outputs(cache_path, logCluL)

        if self.logmain:
            self._append_main_output(cache_path, logCluL)

        try:
            os.remove(cache_path)
        except OSError:
            pass

        logging.info(f'rootNodePath: {rootNodePath}')
        with open(rootNodePath, 'wb') as output:
            pickle.dump(rootNode, output, pickle.HIGHEST_PROTOCOL)
        logging.info(f'logCluLPath: {logCluLPath}')
        with open(logCluLPath, 'wb') as output:
            pickle.dump(logCluL, output, pickle.HIGHEST_PROTOCOL)
        logging.info('Store objects done.')

        logging.info('Parsing done. [Time taken: {!s}]'.format(datetime.now() - starttime))

    def outputResult(self, logClustL):
        if self.df_log.shape[0] == 0:
            return

        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        df_event = []

        for logclust in logClustL:
            template_str = ' '.join(logclust.logTemplate)
            eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logid in logclust.logIDL:
                if logid <= self.lastestLineId:
                    continue
                templates[logid - self.lastestLineId - 1] = template_str
                ids[logid - self.lastestLineId - 1] = eid
            df_event.append([eid, template_str, len(logclust.logIDL)])

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self.get_parameter_list, axis=1)
        logging.info('Output parse file')
        self.df_log.to_csv(os.path.join(self.savePath, self.logname + '_structured.csv'), index=False)
        df_event.to_csv(os.path.join(self.savePath, self.logname + '_templates.csv'), index=False)

        # output Main file
        if self.logmain:
            if not os.path.exists(os.path.join(self.savePath, self.logmain + '_main_structured.csv')):
                logging.info('Output main file for append')
                self.df_log.to_csv(os.path.join(self.savePath, self.logmain + '_main_structured.csv'), index=False)
                df_event.to_csv(os.path.join(self.savePath, self.logmain + '_main_templates.csv'), index=False)

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.logformat)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logname), regex, headers, self.logformat)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe
        """
        log_messages = []
        linecount = 0
        total_line = self._count_lines(log_file)

        with open(log_file, 'r') as fin:
            for line in fin:
                if len(line) > self.text_max_length:
                    logging.error('Length of log string is too long')
                    logging.error(line)
                    continue
                if self.date_filter not in line:
                    # logging.warning(f'{self.date_filter} is not in {line}')
                    continue
                signal.signal(signal.SIGALRM, self._log_to_dataframe_handler)
                signal.alarm(1)
                line = re.sub(r'[^\x00-\x7F]+', '<NASCII>', line)
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                    if linecount % 10000 == 0 or linecount == total_line:
                        logging.info('Loaded {0:.1f}% of log lines.'.format(linecount*100/total_line))
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
            logging.error(e)
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
        logging.error("_get_parameter_list function is hangs!")
        raise Exception("TIME OUT!")

    def _log_to_dataframe_handler(self, signum, frame):
        logging.error('log_to_dataframe function is hangs')
        raise Exception("TIME OUT!")

    def appendResult(self, logClustL):
        if self.df_log is None or self.df_log.shape[0] == 0:
            return
        if self.logmain is None:
            return

        main_structured_path = os.path.join(self.savePath, self.logmain + '_main_structured.csv')
        main_exists = os.path.exists(main_structured_path)
        if main_exists:
            df_log_main_structured = pd.read_csv(main_structured_path, usecols=['LineId'])
            lastestLineId = df_log_main_structured['LineId'].max()
            logging.info(f'lastestLineId: {lastestLineId}')
        else:
            lastestLineId = 0

        line_to_event, df_event = self._build_event_lookup(logClustL)
        df_new = self.df_log[self.df_log['LineId'] > lastestLineId].copy()
        df_new['EventId'] = df_new['LineId'].map(lambda line_id: line_to_event.get(line_id, (0, 0))[0])
        df_new['EventTemplate'] = df_new['LineId'].map(lambda line_id: line_to_event.get(line_id, (0, 0))[1])
        if self.keep_para:
            df_new['ParameterList'] = df_new.apply(self.get_parameter_list, axis=1)

        df_new = df_new[df_new['EventId'] != 0]
        if not df_new.empty:
            df_new.to_csv(
                main_structured_path,
                index=False,
                mode='a' if main_exists else 'w',
                header=not main_exists,
            )
        df_event.to_csv(os.path.join(self.savePath, self.logmain + '_main_templates.csv'), index=False)

    def _build_event_lookup(self, logClustL):
        line_to_event = {}
        df_event = []

        for logclust in logClustL:
            template_str = ' '.join(logclust.logTemplate)
            eid = hashlib.md5(template_str.encode('utf-8')).hexdigest()[0:8]
            for logid in logclust.logIDL:
                if logid <= self.lastestLineId:
                    continue
                line_to_event[logid] = (eid, template_str)
            df_event.append([eid, template_str, len(logclust.logIDL)])

        return line_to_event, pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

    def _write_outputs(self, cache_path, logClustL):
        if self._count_lines(cache_path) == 0:
            return

        line_to_event, df_event = self._build_event_lookup(logClustL)
        structured_path = os.path.join(self.savePath, self.logname + '_structured.csv')
        self._write_structured_csv(cache_path, structured_path, line_to_event, append=False)
        df_event.to_csv(os.path.join(self.savePath, self.logname + '_templates.csv'), index=False)

    def _append_main_output(self, cache_path, logClustL):
        if self.logmain is None or self._count_lines(cache_path) == 0:
            return

        line_to_event, df_event = self._build_event_lookup(logClustL)
        main_structured_path = os.path.join(self.savePath, self.logmain + '_main_structured.csv')
        self._write_structured_csv(cache_path, main_structured_path, line_to_event, append=True)
        df_event.to_csv(os.path.join(self.savePath, self.logmain + '_main_templates.csv'), index=False)

    def _write_structured_csv(self, cache_path, output_path, line_to_event, append=False):
        first_chunk = not append or not os.path.exists(output_path)
        for chunk in pd.read_csv(cache_path, chunksize=10000):
            event_info = chunk['LineId'].map(line_to_event)
            chunk['EventId'] = event_info.apply(lambda x: x[0] if isinstance(x, tuple) else 0)
            chunk['EventTemplate'] = event_info.apply(lambda x: x[1] if isinstance(x, tuple) else 0)
            if self.keep_para:
                chunk['ParameterList'] = chunk.apply(self.get_parameter_list, axis=1)
            chunk.to_csv(output_path, index=False, mode='a' if not first_chunk else 'w', header=first_chunk)
            first_chunk = False
