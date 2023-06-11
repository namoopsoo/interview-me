from pathlib import Path
from itertools import chain


def build_texts_from_repository(repo_dir):
    """Return a dataset of the code
    """
    dataset = []
    file_types = []
    for path in chain(
        Path(repo_dir).glob("**/*.py"),
        Path(repo_dir).glob("**/*.md"),
    ):
        assert path.is_file() and path.suffix
        lines = path.read_text().splitlines()
        
        dataset.extend(
            [[{"line_number": i,
               "line": line,
               "path": str(path.relative_to(repo_dir))}]
        for i, line in enumerate(lines)
         ]
        )
    return dataset
