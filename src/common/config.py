"""
Based on from https://github.com/kelvinguu/lang2program/blob/master/third-party/gtd/gtd/utils.py
"""

import json
from copy import deepcopy

from pyhocon import ConfigTree, HOCONConverter, ConfigFactory


class HoconConfig:
    """A wrapper around the pyhocon ConfigTree object.

    Allows you to access values in the ConfigTree as attributes.
    """

    def __init__(self, config_tree: ConfigTree = None):
        """Create a Config.

        Args:
            config_tree (ConfigTree)
        """
        if config_tree is None:
            config_tree = ConfigTree()
        self._config_tree = config_tree

    def __getattr__(self, item: str):
        if item in [
            "trainable_weights",
            "non_trainable_weights",
            "_use_resource_variables",
            "weakrefself_target__",
            "_tf_decorator",
        ]:
            raise AttributeError()

        val = self._config_tree[item]
        if isinstance(val, ConfigTree):
            return HoconConfig(val)
        else:
            return val

    def get(self, key, default=None):
        val = self._config_tree.get(key, default)
        if isinstance(val, ConfigTree):
            return HoconConfig(val)
        else:
            return val

    def put(self, key, value, append=False):
        """Put a value into the Config (dot separated)

        Args:
            key (str): key to use (dot separated). E.g. `a.b.c`
            value (object): value to put
        """
        self._config_tree.put(key, value, append=append)

    def __repr__(self):
        return self.to_str()

    def to_str(self):
        return HOCONConverter.convert(self._config_tree, "hocon")

    def to_json(self):
        return json.loads(HOCONConverter.convert(self._config_tree, "json"))

    def to_dict(self):
        return self.to_json()

    def to_file(self, path, type="hocon"):
        with open(path, "w") as f:
            if type == "hocon":
                f.write(self.to_str())
            elif type == "json":
                f.write(json.dumps(self.to_dict(), indent=4))

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return HoconConfig(deepcopy(self._config_tree))

    @classmethod
    def from_file(cls, path):
        config_tree = ConfigFactory.parse_file(path)
        return cls(config_tree)

    @classmethod
    def from_str(cls, s):
        config_tree = ConfigFactory.parse_string(s)
        return cls(config_tree)

    @classmethod
    def from_dict(cls, d):
        return HoconConfig(ConfigFactory.from_dict(d))

    @classmethod
    def merge(cls, configs):
        for c in configs:
            assert isinstance(c, HoconConfig)

        ctree = configs[0]._config_tree
        for c in configs[1:]:
            ctree = ConfigTree.merge_configs(ctree, c._config_tree)

        return cls(ctree)

    @classmethod
    def merge_to_new(cls, configs):
        for i in range(len(configs)):
            assert isinstance(configs[i], HoconConfig)
            configs[i] = deepcopy(configs[i])

        ctree = configs[0]._config_tree
        for c in configs[1:]:
            ctree = ConfigTree.merge_configs(ctree, c._config_tree)

        return cls(ctree)

    @classmethod
    def from_files(cls, paths):
        configs = [HoconConfig.from_file(p) for p in paths]
        return HoconConfig.merge(configs)  # later configs overwrite earlier configs

    def to_class(self, clazz):
        return cattr.structure(self.to_dict(), clazz)
