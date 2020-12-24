from argparse import ArgumentParser


class Configurable():
    classes = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Configurable.classes.add(cls)

    def __init__(self, conf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conf = conf


class AutoArgParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        added_names = set()
        for cls in Configurable.classes:
            for arg in getattr(cls, 'args', []):
                name, kwargs = arg
                if name in added_names:
                    continue
                self.add_argument(name, **kwargs)
                added_names.add(name)
