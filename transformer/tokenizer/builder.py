"""Simple BPE Tokenizer builder."""

from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

class TokenizerBuilder:
    """Creates BPE tokenizers."""

    def __init__(self,vocab_size: int = 30000,min_frequency: int = 2,unk_token: str = "[UNK]",special_tokens: list[str] | None = None,):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.unk_token = unk_token
        self.special_tokens = special_tokens or ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]

    def _get_all_sequences(self, ds, lang: str):
        for item in ds:
            yield item["translation"][lang]

    def _create_tokenizer(self):
        tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        tokenizer.pre_tokenizer = Whitespace()
        return tokenizer

    def _create_trainer(self):
        return BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
        )

    def build_or_load(self, config, ds, lang):
        """Build or load a tokenizer."""
        tokenizer_path = Path(config["tokenizer_file"].format(lang))

        if not tokenizer_path.exists():
            tokenizer = self._create_tokenizer()
            trainer = self._create_trainer()
            tokenizer.train_from_iterator(
                self._get_all_sequences(ds, lang), trainer=trainer
            )
            tokenizer.save(str(tokenizer_path))
        else:
            tokenizer = Tokenizer.from_file(str(tokenizer_path))

        return tokenizer
