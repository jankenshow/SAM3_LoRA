"""Build hook for the upstream vocabulary; metadata lives in pyproject.toml."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    """Bundle the vocabulary beside sam3 in wheels without modifying the submodule."""

    def vocabulary_output(self):
        return str(Path(self.build_lib) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")

    def run(self):
        source = Path(__file__).parent / "vendor/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        if not source.is_file():
            raise FileNotFoundError(
                "Initialize SAM3 before building: git submodule update --init --recursive"
            )
        super().run()
        destination = self.vocabulary_output()
        self.mkpath(str(Path(destination).parent))
        self.copy_file(str(source), destination)

    def get_outputs(self, include_bytecode=1):
        return super().get_outputs(include_bytecode) + [self.vocabulary_output()]


setup(cmdclass={"build_py": BuildPy})
