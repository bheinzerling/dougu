from argparse import ArgumentParser


class Configurable():
    classes = set()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Configurable.classes.add(cls)

    def __init__(self, conf, *args, **kwargs):
        super().__init__()
        self.conf = conf

    def arg_keys(self):
        return [
            arg[0][2:].replace('-', '_') for arg in getattr(self, 'args', [])
            ]


class AutoArgParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        added_names = dict()
        for cls in Configurable.classes:
            for arg in getattr(cls, 'args', []):
                name, kwargs = arg
                if name in added_names:
                    other_cls, other_kwargs = added_names[name]
                    if kwargs != other_kwargs:
                        raise ValueError(
                            f'Argument conflict. Argument "{name}" exists '
                            f'in {other_cls} with options {other_kwargs} '
                            f'and in {cls} with options {kwargs}')
                    else:
                        continue
                self.add_argument(name, **kwargs)
                added_names[name] = (cls, kwargs)
