#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr
# Pavel KORSHUNOV - pavel.korshunov@idiap.ch

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import os
from pyannote.core import Segment, Timeline, Annotation, SlidingWindow
from pyannote.database import Database
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.database.protocol import SpeakerSpottingProtocol
from pandas import read_table
from pathlib import Path


# this protocol defines a speaker diarization protocol: as such, a few methods
# needs to be defined: trn_iter, dev_iter, and tst_iter.

class SpeakerDiarization(SpeakerDiarizationProtocol):
    """My first speaker diarization protocol """

    def _load_data(self, sre, subset):

        data_dir = Path(__file__).parent / 'data' / 'speaker_diarization'

        annotated = data_dir / f'{sre}-{subset}.uem'
        names = ['uri', 'NA0', 'start', 'end']
        annotated = read_table(annotated, delim_whitespace=True, names=names)

        annotation = data_dir / f'{sre}-{subset}.mdtm'
        names = ['uri', 'channel', 'start', 'duration',
                 'NA1', 'NA2', 'gender', 'speaker']
        annotation = read_table(annotation, delim_whitespace=True, names=names)

        return {'annotated': annotated,
                'annotation': annotation}

    def _xxx_iter(self, sre, subsets):
        for subset in subsets:
            data = self._load_data(sre, subset)

            AnnotatedGroups = data['annotated'].groupby(by='uri')
            AnnotationGroups = data['annotation'].groupby(by='uri')

            for uri, annotated in AnnotatedGroups:

                segments = []
                for segment in annotated.itertuples():
                    segments.append(Segment(start=segment.start, end=segment.end))

                annotation = Annotation(uri=uri)
                channel = None

                sample_names = AnnotationGroups.groups.keys()
                # process only those uris that we have inside annotations
                if uri in sample_names:
                    for t, turn in enumerate(AnnotationGroups.get_group(uri).itertuples()):
                        if t == 0:
                            channel = 0 if turn.channel == 'A' else 1
                        segment = Segment(start=turn.start,
                                          end=turn.start + turn.duration)
                        annotation[segment, t] = turn.speaker

                    current_file = {
                        'database': 'SRE',
                        'uri': uri,
                        'channel': channel,
                        'annotated': Timeline(uri=uri, segments=segments),
                        'annotation': annotation}
                    yield current_file


    def trn_iter(self):
        # no train set
        pass

    def dev_iter(self):
        return self._xxx_iter('sre08', ['engtel-models', 'test-summed'])

    def tst_iter(self):
        return self._xxx_iter('sre10', ['engtel-models', 'test-summed'])


class SpeakerSpotting(SpeakerDiarization, SpeakerSpottingProtocol):

    def _sessionify(self, current_files):

        for current_file in current_files:

            annotated = current_file['annotated']
            annotation = current_file['annotation']

            for segment in annotated:
                sessions = SlidingWindow(start=segment.start,
                                         duration=60., step=60.,
                                         end=segment.end - 60.)

                for session in sessions:

                    session_file = dict(current_file)
                    session_file['annotated'] = annotated.crop(session)
                    session_file['annotation'] = annotation.crop(session)

                    yield session_file

    def trn_iter(self):
        # no train set
        pass

    def dev_iter(self):
        return self._sessionify(super().dev_iter())

    def tst_iter(self):
        return self._sessionify(super().tst_iter())

    def _xxx_enrol_iter(self, sre):

        # load enrolments
        data_dir = Path(__file__).parent / 'data' / 'speaker_spotting'
        enrolments = data_dir / f'{sre}-engtel-models.mdtm'
        names = ['uri', 'channel', 'start', 'duration',
                 'NA1', 'NA2', 'NA3', 'model_id']
        enrolments = read_table(enrolments, delim_whitespace=True, names=names)

        AnnotationGroups = enrolments.groupby(by=['uri', 'model_id'])
        for grouped_tuple, turns in AnnotationGroups:
            uri = grouped_tuple[0]
            model_id = grouped_tuple[1]

            # gather enrolment data
            segments = []
            channel = None
            for t, turn in enumerate(turns.itertuples()):
                if t == 0:
                    channel = 0 if turn.channel == 'A' else 1
                segment = Segment(start=turn.start,
                                  end=turn.start + turn.duration)
                if segment:
                    segments.append(segment)
            enrol_with = Timeline(segments=segments, uri=uri)

            current_enrolment = {
                'database': 'SRE',
                'uri': uri,
                'channel': channel,
                'model_id': model_id,
                'enrol_with': enrol_with,
            }

            yield current_enrolment

    def dev_enrol_iter(self):
        return self._xxx_enrol_iter('sre08')

    def tst_enrol_iter(self):
        return self._xxx_enrol_iter('sre10')

    def dev_try_iter(self):
        # no development set
        pass


    def _xxx_try_iter(self, sre):

        # load "who speaks when" reference
        data = self._load_data(sre, 'test-summed')

        diarization = getattr(self, 'diarization', True)
        if diarization:
            AnnotationGroups = data['annotation'].groupby(by='uri')
        else:
            AnnotationGroups = data['annotation'].groupby(by=['uri', 'speaker'])

        # load trials
        data_dir = Path(__file__).parent / 'data' / 'speaker_spotting'
        trials = data_dir / f'{sre}-engtel-trials.txt'
        names = ['model_id', 'uri', 'start', 'end', 'target', 'first', 'total']
        trials = read_table(trials, delim_whitespace=True, names=names)

        for trial in trials.itertuples():

            model_id = trial.model_id
            speaker = model_id
            uri = trial.uri

            # trial session
            try_with = Segment(start=trial.start, end=trial.end)

            if diarization:
                # 'annotation' & 'annotated' are needed when diarization is set
                # therefore, this needs a bit more work than when set to False.

                annotation = Annotation(uri=uri)
                turns = AnnotationGroups.get_group(uri)
                for t, turn in enumerate(turns.itertuples()):
                    segment = Segment(start=turn.start,
                                      end=turn.start + turn.duration)
                    if not (segment & try_with):
                        continue
                    annotation[segment, t] = turn.speaker

                annotation = annotation.crop(try_with)
                reference = annotation.label_timeline(speaker)
                annotated = Timeline(uri=uri, segments=[try_with])

                # pack & yield trial
                current_trial = {
                    'database': 'SRE',
                    'uri': uri,
                    'try_with': try_with,
                    'model_id': model_id,
                    'reference': reference,
                    'annotation': annotation,
                    'annotated': annotated,
                }

            else:
                # 'annotation' & 'annotated' are not needed when diarization is
                # set to False -- leading to a faster implementation...
                segments = []
                if trial.target == 'target':
                    turns = AnnotationGroups.get_group((uri, speaker))
                    for t, turn in enumerate(turns.itertuples()):
                        segment = Segment(start=turn.start,
                                          end=turn.start + turn.duration)
                        segments.append(segment)
                reference = Timeline(uri=uri, segments=segments).crop(try_with)

                # pack & yield trial
                current_trial = {
                    'database': 'SRE',
                    'uri': uri,
                    'try_with': try_with,
                    'model_id': model_id,
                    'reference': reference,
                }

            yield current_trial

    def tst_try_iter(self):
        def get_turns(uri):
            ref_file_path = Path(__file__).parent / 'data' / 'speaker_diarization' / uri
            ref_file_path = Path(str(ref_file_path) + '.txt')
            gt_names = ['start', 'end', 'speaker', 'speakerID']
            return read_table(os.path.join(data_dir, ref_file_path), delim_whitespace=True, names=gt_names)

        diarization = getattr(self, 'diarization', True)

        # load trials
        data_dir = Path(__file__).parent / 'data' / 'speaker_spotting'
        trials = data_dir / 'tst.trial.txt'
        names = ['model_id', 'uri', 'start', 'end', 'target', 'first', 'total']
        trials = read_table(trials, delim_whitespace=True, names=names)

        for trial in trials.itertuples():

            model_id = trial.model_id

            speaker = model_id

            uri = trial.uri

            # trial session
            try_with = Segment(start=trial.start, end=trial.end)

            if diarization:
                # 'annotation' & 'annotated' are needed when diarization is set
                # therefore, this needs a bit more work than when set to False.

                annotation = Annotation(uri=uri)
                turns = get_turns(uri)
                for t, turn in enumerate(turns.itertuples()):
                    segment = Segment(start=turn.start,
                                      end=turn.end)
                    if not (segment & try_with):
                        continue
                    annotation[segment, t] = turn.speakerID

                annotation = annotation.crop(try_with)
                reference = annotation.label_timeline(speaker)
                annotated = Timeline(uri=uri, segments=[try_with])

                # pack & yield trial
                current_trial = {
                    'database': 'SRE',
                    'uri': uri,
                    'try_with': try_with,
                    'model_id': model_id,
                    'reference': reference,
                    'annotation': annotation,
                    'annotated': annotated,
                }

            else:
                # 'annotation' & 'annotated' are not needed when diarization is
                # set to False -- leading to a faster implementation...
                segments = []
                if trial.target == 'target':
                    turns = get_turns(uri).groupby(by='speakerID')
                    for t, turn in enumerate(turns.get_group(speaker).itertuples()):
                        segment = Segment(start=turn.start,
                                          end=turn.end)
                        segments.append(segment)
                reference = Timeline(uri=uri, segments=segments).crop(try_with)

                # pack & yield trial
                current_trial = {
                    'database': 'SRE',
                    'uri': uri,
                    'try_with': try_with,
                    'model_id': model_id,
                    'reference': reference,
                }

            yield current_trial


# this is where we define each protocol for this database.
# without this, `pyannote.database.get_protocol` won't be able to find them...

class SRE(Database):
    """SRE database by NIST: dev set SR08 and test set SRE10  """

    def __init__(self, preprocessors={}, **kwargs):
        super(SRE, self).__init__(preprocessors=preprocessors, **kwargs)

        # register the first protocol: it will be known as
        self.register_protocol(
            'SpeakerDiarization', 'Fullset', SpeakerDiarization)

        self.register_protocol(
            'SpeakerSpotting', 'Fullset', SpeakerSpotting)
