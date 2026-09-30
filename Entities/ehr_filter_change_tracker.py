import json

class FilterChangeTrackerEHR:
    """Counts what one filter did to one admission, at three levels of severity.

    The three levels answer "how destructive was this filter", which is the whole
    reason we track it — a filter that nudges a few values is a different
    proposition from one that throws out whole patients:

      value_level    individual measurements changed or dropped
      feature_level  an entire vital wiped out for a record
      sample_level   the record itself dropped from the dataset

    They're independent counters, not nested, so a record can register at more
    than one level.
    """
    filter_name: str
    admission_id: int
    value_level: int
    feature_level: int
    sample_level: int

    def __init__(self, filter_name: str, admission_id: int):
        self.filter_name = filter_name
        self.admission_id = admission_id
        self.value_level = 0
        self.feature_level = 0
        self.sample_level = 0

    def __add__(self, other):
        """Summing trackers rolls per-record counts up to a per-filter total.

        The `other == 0` check plus __radd__ is what makes the builtin sum() work
        on a list of these — sum() starts from 0, and without this the very first
        addition explodes. Result keeps the first operand's filter_name and
        admission_id, so a summed tracker's admission_id is meaningless; only the
        three counters are.
        """
        if other == 0:
            return self

        new_tracker = FilterChangeTrackerEHR(self.filter_name, self.admission_id)
        new_tracker.value_level = self.value_level + other.value_level
        new_tracker.feature_level = self.feature_level + other.feature_level
        new_tracker.sample_level = self.sample_level + other.sample_level
        return new_tracker

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def to_dict(self):
        return self.__dict__

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self.__dict__, file, indent=4)

    @classmethod
    def load(cls, path: str):
        # Dead since save() stopped being used — everything writes lists via
        # save_all now. Not a drop-in swap: load_all returns a list, this
        # returned a single tracker, so callers need adjusting rather than
        # renaming. Raising instead of silently returning the wrong shape.
        raise DeprecationWarning("This method is deprecated. Please use `load_all` instead. NOTE: `load_all` is not a drop in replacement for `load`.")

    @classmethod
    def save_all(cls, trackers, path: str):
        # getattr-with-default rather than direct access because older pickled
        # trackers predate one of the three counters. Same reason load_all uses
        # .get() on the way back in.
        data_list = []
        for tracker in trackers:
            data_list.append({
                'filter_name': tracker.filter_name,
                'admission_id': tracker.admission_id,
                'value_level': getattr(tracker, 'value_level', 0),
                'feature_level': getattr(tracker, 'feature_level', 0),
                'sample_level': getattr(tracker, 'sample_level', 0)
            })
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(data_list, file, indent=4)

    @classmethod
    def load_all(cls, path: str):
        with open(path, 'r', encoding='utf-8') as file:
            data_list = json.load(file)

        trackers = []
        for data in data_list:
            tracker = cls(data['filter_name'], data['admission_id'])
            tracker.value_level = data.get('value_level', 0)
            tracker.feature_level = data.get('feature_level', 0)
            tracker.sample_level = data.get('sample_level', 0)
            trackers.append(tracker)

        return trackers
