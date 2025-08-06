

import torch
import warnings

class NiceRepr(object):

    def __nice__(self):
        if hasattr(self, "__len__"):
            return str(len(self))
        else:
            raise NotImplementedError(
                f"Define the __nice__ method for {self.__class__!r}"
            )

    def __repr__(self):
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f"<{classname}({nice}) at {hex(id(self))}>"
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f"<{classname}({nice})>"
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

class AssignResult(NiceRepr):

    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels
        self._extra_properties = {}

    @property
    def num_preds(self):
        return len(self.gt_inds)

    def set_extra_property(self, key, value):
        assert key not in self.info
        self._extra_properties[key] = value

    def get_extra_property(self, key):
        return self._extra_properties.get(key, None)

    @property
    def info(self):
        basic_info = {
            "num_gts": self.num_gts,
            "num_preds": self.num_preds,
            "gt_inds": self.gt_inds,
            "max_overlaps": self.max_overlaps,
            "labels": self.labels,
        }
        basic_info.update(self._extra_properties)
        return basic_info

    def __nice__(self):
        parts = []
        parts.append(f"num_gts={self.num_gts!r}")
        if self.gt_inds is None:
            parts.append(f"gt_inds={self.gt_inds!r}")
        else:
            parts.append(f"gt_inds.shape={tuple(self.gt_inds.shape)!r}")
        if self.max_overlaps is None:
            parts.append(f"max_overlaps={self.max_overlaps!r}")
        else:
            parts.append("max_overlaps.shape=" f"{tuple(self.max_overlaps.shape)!r}")
        if self.labels is None:
            parts.append(f"labels={self.labels!r}")
        else:
            parts.append(f"labels.shape={tuple(self.labels.shape)!r}")
        return ", ".join(parts)

    @classmethod
    def random(cls, **kwargs):
        rng = kwargs.get("rng", None)
        num_gts = kwargs.get("num_gts", None)
        num_preds = kwargs.get("num_preds", None)
        p_ignore = kwargs.get("p_ignore", 0.3)
        p_assigned = kwargs.get("p_assigned", 0.7)
        p_use_label = kwargs.get("p_use_label", 0.5)
        num_classes = kwargs.get("p_use_label", 3)

        import numpy as np

        if rng is None:
            rng = np.random.mtrand._rand
        elif isinstance(rng, int):
            rng = np.random.RandomState(rng)
        else:
            rng = rng
        if num_gts is None:
            num_gts = rng.randint(0, 8)
        if num_preds is None:
            num_preds = rng.randint(0, 16)

        if num_gts == 0:
            max_overlaps = torch.zeros(num_preds, dtype=torch.float32)
            gt_inds = torch.zeros(num_preds, dtype=torch.int64)
            if p_use_label is True or p_use_label < rng.rand():
                labels = torch.zeros(num_preds, dtype=torch.int64)
            else:
                labels = None
        else:
            import numpy as np

            max_overlaps = torch.from_numpy(rng.rand(num_preds))

            is_assigned = torch.from_numpy(rng.rand(num_preds) < p_assigned)
            n_assigned = min(num_preds, min(num_gts, is_assigned.sum()))

            assigned_idxs = np.where(is_assigned)[0]
            rng.shuffle(assigned_idxs)
            assigned_idxs = assigned_idxs[0:n_assigned]
            assigned_idxs.sort()

            is_assigned[:] = 0
            is_assigned[assigned_idxs] = True

            is_ignore = torch.from_numpy(rng.rand(num_preds) < p_ignore) & is_assigned

            gt_inds = torch.zeros(num_preds, dtype=torch.int64)

            true_idxs = np.arange(num_gts)
            rng.shuffle(true_idxs)
            true_idxs = torch.from_numpy(true_idxs)
            gt_inds[is_assigned] = true_idxs[:n_assigned]

            gt_inds = torch.from_numpy(rng.randint(1, num_gts + 1, size=num_preds))
            gt_inds[is_ignore] = -1
            gt_inds[~is_assigned] = 0
            max_overlaps[~is_assigned] = 0

            if p_use_label is True or p_use_label < rng.rand():
                if num_classes == 0:
                    labels = torch.zeros(num_preds, dtype=torch.int64)
                else:
                    labels = torch.from_numpy(
                        rng.randint(0, num_classes, size=num_preds)
                    )
                    labels[~is_assigned] = 0
            else:
                labels = None

        self = cls(num_gts, gt_inds, max_overlaps, labels)
        return self

    def add_gt_(self, gt_labels):
        self_inds = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=gt_labels.device
        )
        self.gt_inds = torch.cat([self_inds, self.gt_inds])

        self.max_overlaps = torch.cat(
            [self.max_overlaps.new_ones(len(gt_labels)), self.max_overlaps]
        )

        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])
