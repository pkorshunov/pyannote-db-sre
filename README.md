# NIST SRE database plugin for pyannote.database

## Installation

```bash
$ pip install pyannote.db.sre  # install from pip, or
$ pip install -e .  # install a local copy
```

Tell `pyannote` where to look for NIST SRE audio files. 
Note that both SRE08 and SRE10 databases should be located in the same foldder.

```bash
$ cat ~/.pyannote/db.yml
SRE: /path/to/nist_sre/{uri}.sph
```

SRE database has Dev set from SRE08 data and Test set from SRE10 data.

## Speaker diarization protocol


Protocol is initialized as follows:

```python
>>> from pyannote.database import get_protocol, FileFinder
>>> preprocessors = {'audio': FileFinder()}
>>> protocol = get_protocol('SRE.SpeakerDiarization.Fullset',
...                         preprocessors=preprocessors)
```

### Test / Evaluation

```python
>>> # initialize evaluation metric
>>> from pyannote.metrics.diarization import DiarizationErrorRate
>>> metric = DiarizationErrorRate()
>>>
>>> # iterate over each file of the test set
>>> for test_file in protocol.test():
...
...     # process test file
...     audio = test_file['audio']
...     hypothesis = process_file(audio)
...
...     # evaluate hypothesis
...     reference = test_file['annotation']
...     uem = test_file['annotated']
...     metric(reference, hypothesis, uem=uem)
>>>
>>> # report results
>>> metric.report(display=True)
```

## Speaker spotting procotol

Protocol is initialized as follows:

```python
>>> from pyannote.database import get_protocol, FileFinder
>>> preprocessors = {'audio': FileFinder()}
>>> protocol = get_protocol('SRE.SpeakerSpotting.Fullset',
...                         preprocessors=preprocessors)
```

### Enrolment

```python
>>> # dictionary meant to store all target models
>>> models = {}
>>>
>>> # iterate over all enrolments
>>> for current_enrolment in protocol.test_enrolment():
...
...     # target identifier
...     target = current_enrolment['model_id']
...     # the same speaker may be enrolled several times using different target
...     # identifiers. in other words, two different identifiers does not
...     # necessarily not mean two different speakers.
...
...     # path to audio file to use for enrolment
...     audio = current_enrolment['audio']
...
...     # pyannote.core.Timeline containing target speech turns
...     # See http://pyannote.github.io/pyannote-core/structure.html#timeline
...     enrol_with = current_enrolment['enrol_with']
...
...     # this is where enrolment actually happens and model is stored
...     models[target] = enrol(audio, enrol_with)
```

The following pseudo-code shows what the `enrol` function could look like:

```python
>>> def enrol(audio, enrol_with):

```