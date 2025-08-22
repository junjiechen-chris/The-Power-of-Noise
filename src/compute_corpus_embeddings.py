import os 
import argparse
import warnings

import torch
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset, Features, Value

from retriever import *
from utils import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
SEED=10
# force TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
features = Features({"text": Value("string"), "title": Value("string"),})


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Script for computing the embeddings of a corpus.")
    parser.add_argument('--corpus_path', type=str, help='Path to the JSON corpus data')
    parser.add_argument('--encoder_id', type=str, default='facebook/contriever', help='Model identifier for the encoder')
    parser.add_argument('--max_length_encoder', type=int, default=512, help='Maximum sequence length for the encoder')
    parser.add_argument('--normalize_embeddings', type=str2bool, default=False, help='Whether to normalize embeddings')
    parser.add_argument('--lower_case', type=str2bool, default=False, help='Whether to lower case the corpus text')
    parser.add_argument('--do_normalize_text', type=str2bool, default=True, help='Whether to normalize the corpus text')
    parser.add_argument('--output_dir', type=str, default='data/corpus/embeddings/', help='Output directory for saving embeddings')
    parser.add_argument('--prefix_name', type=str, default='contriever', help='Initial part of the name of the saved embeddings')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for embedding documents')
    parser.add_argument('--save_every', type=int, default=500)
    
    args =  parser.parse_args()

    return args


def initialize_retriever(args: argparse.Namespace) -> Retriever:
    """Initialize the encoder and retriever."""
    config = AutoConfig.from_pretrained(args.encoder_id)
    encoder = Encoder(config).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_id)
    retriever = Retriever(
        device=device, tokenizer=tokenizer, 
        query_encoder=encoder, 
        max_length=args.max_length_encoder,
        norm_doc_emb=args.normalize_embeddings,
        lower_case=args.lower_case,
        do_normalize_text=args.do_normalize_text
    )

    return retriever



def read_json_streaming(file_path: str):
    """Streaming version using ijson"""
    with open(file_path, "rb") as reader:
        # Assumes JSON structure is an array of objects: [{"title": "...", "text": "..."}, ...]
        parser = ijson.parse(reader)
        for item in ijson.items(reader, 'item'):
            yield item

def process_text(item: Dict[str, str], do_normalize_text, lower_case) -> str:
    """Process individual item (same logic as original)"""
    # Concatenate title and text
    text = item["title"] + " " + item["text"] if len(item["title"]) > 0 else item["text"]
    
    # Apply text normalization if enabled
    if do_normalize_text:
        text = normalize_text.normalize(text)
        
    # Apply lowercasing if enabled
    if lower_case:
        text = text.lower()
            
    return {"plain": text}


class CorpusDataset(IterableDataset):
    """
    Enhanced CorpusDataset that supports both in-memory list and streaming from file.
    Minimal change from original - just pass file_path instead of corpus_info for streaming.
    """
    def __init__(
        self, 
        corpus_info,
        lower_case: bool = False,
        do_normalize_text: bool = False
    ):
        self.corpus_info = corpus_info
        self.lower_case = lower_case
        self.do_normalize_text = do_normalize_text
        self.is_streaming = isinstance(corpus_info, str)  # If string, treat as file path
        
    def _process_item(self, item: Dict[str, str]) -> str:
        """Process individual item (same logic as original)"""
        # Concatenate title and text
        text = item["title"] + " " + item["text"] if len(item["title"]) > 0 else item["text"]
        
        # Apply text normalization if enabled
        if self.do_normalize_text:
            text = normalize_text.normalize(text)
        
        # Apply lowercasing if enabled
        if self.lower_case:
            text = text.lower()
            
        return text
    
    def __iter__(self):
        if self.is_streaming:
            # Streaming mode: corpus_info is file path
            with open(self.corpus_info, "rb") as reader:
                # Parse JSON array items one by one
                for item in ijson.items(reader, 'item'):
                    yield self._process_item(item)
        else:
            # Original mode: corpus_info is list
            for item in self.corpus_info:
                yield self._process_item(item)
    
    def __len__(self):
        if self.is_streaming:
            # For streaming, we can't know length without reading entire file
            # Return None or raise NotImplementedError
            raise NotImplementedError("Length not available for streaming datasets")
        else:
            return len(self.corpus_info)


def main():
    args = parse_arguments()

    print("Loading corpus...")
    # corpus = CorpusDataset(args.corpus_path, lower_case=args.lower_case, do_normalize_text=args.do_normalize_text)

    corpus = load_dataset("json", data_files=args.corpus_path, split="train", streaming=True, features=features)
    corpus = corpus.map(lambda x: process_text(x, lower_case=args.lower_case, do_normalize_text=args.do_normalize_text))
    
    print("Corpus loaded")

    retriever = initialize_retriever(args)
    print("Computing embeddings...")
    retriever.encode_corpus(
        corpus, 
        batch_size=args.batch_size, 
        output_dir=args.output_dir,
        prefix_name=args.prefix_name,
        save_every=args.save_every
    )

if __name__ == '__main__':
    seed_everything(SEED)
    main()