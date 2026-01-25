class SimpleTokenizer:

    def __init__(self, text: str) -> None:
        self.text = text
        self.vocab = sorted(list(set(text)))
        self.encoder_map = {ch:i for i, ch in enumerate(self.vocab)}
        self.decoder_map = {i:ch for i, ch in enumerate(self.vocab)}

    def encode(self, text):
       return [self.encoder_map[x] for x in text] 
    
    def decode(self, text):
        return [self.decoder_map[x] for x in text] 