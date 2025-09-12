import os 
import hydra
import warnings
import logging
from tqdm import tqdm
from typing import Tuple, Dict, Optional, Union, List
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from llm import LLM
from vllm_wrapper import VLLMWrapper
from vllm_server_client import VLLMServerClient
from transformers import AutoTokenizer
from utils import *
from prompt_dataset import PromptDataset
from datasets import load_dataset
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from dotenv import load_dotenv

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

# Register custom resolver


def validate_config(cfg: DictConfig):
    """Validate the Hydra configuration."""
    if cfg.generation.num_documents_in_context is None:
        raise ValueError("'num_documents_in_context' must be specified.")
    if cfg.generation.num_documents_in_context <= 0:
        raise ValueError("'num_documents_in_context' must be a positive integer.")
    if (cfg.generation.gold_position is not None and 
        (cfg.generation.gold_position < 0 or cfg.generation.gold_position >= cfg.generation.num_documents_in_context)):
        raise ValueError("'gold_position' must be within the range of 'num_documents_in_context'.")


def load_corpus(
    cfg: DictConfig
) -> Tuple[List[Dict], Optional[Dict[int, int]]]:
    # Load the corpus
    if cfg.generation.load_full_corpus:
        corpus = load_dataset('json', data_files=cfg.data.full_corpus_path, split='train')
        return corpus, None
    else:
        corpus, full_to_subset_idx_map = read_subset_corpus_with_config(cfg)
        return corpus, full_to_subset_idx_map


def load_search_results(cfg: DictConfig) -> List[Tuple[List[int], List[float]]]:
    search_results = read_pickle(cfg.corpus.search_results_path)
    return search_results


def initialize_dataset_and_loader(
    cfg: DictConfig, 
    corpus: List[Dict], 
    full_to_subset_idx_map: Optional[Dict[int, int]], 
    search_results: List[Tuple[List[int], List[float]]], 
    tokenizer: PreTrainedTokenizer
) -> DataLoader:
    
    prompt_ds = PromptDataset(
        corpus=corpus, data_path=cfg.data.data_path, 
        tokenizer=tokenizer, 
        max_tokenized_length=cfg.llm.model_max_length - 2, 
        search_results=search_results,
        full_to_subset_idx_map=full_to_subset_idx_map,
        do_normalize_query=True, 
        num_documents_in_context=cfg.generation.num_documents_in_context,
        gold_position=cfg.generation.gold_position,
        get_documents_without_answer=cfg.generation.get_documents_without_answer,
        randomize_gold_position=cfg.generation.randomize_gold_position,
    )
    prompt_dataloader = DataLoader(
        prompt_ds,
        batch_size=cfg.llm.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
    )
    return prompt_dataloader


def print_info(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Configuration:")
    logger.info(f"DATA: {cfg.data.data_path}")
    logger.info(f"MODEL: {cfg.llm.llm_id}")
    logger.info(f"USE RANDOM IN CONTEXT: {cfg.generation.use_random}")
    logger.info(f"GOLD POSITION: {cfg.generation.gold_position}")
    logger.info(f"NUM DOCUMENTS IN CONTEXT: {cfg.generation.num_documents_in_context}")
    logger.info(f"DOCUMENTS WITHOUT ANSWER: {cfg.generation.get_documents_without_answer}")
    logger.info(f"BATCH SIZE: {cfg.llm.batch_size}")
    logger.info(f"SAVE EVERY: {cfg.generation.save_every}")


def generate_and_save(
    cfg: DictConfig, 
    llm: Union[VLLMWrapper, VLLMServerClient],
    prompt_dataloader: DataLoader
):
    logger = logging.getLogger(__name__)
    
    # Info from config
    llm_id = cfg.llm.llm_id
    num_doc = cfg.generation.num_documents_in_context
    save_every = cfg.generation.save_every
    gold_pos = cfg.generation.gold_position
    # retriever_str = "adore" if cfg.generation.use_adore else "contriever"
    rand_str = "_rand" if cfg.generation.use_random else ""
    answerless_str = "_answerless" if cfg.generation.get_documents_without_answer else ""

    # llm_folder = llm_id.split("/")[1] if '/' in llm_id else llm_id
    # saving_dir = f"{hydra_output_dir}/numdoc{num_doc}_gold_at{gold_pos}{rand_str}{answerless_str}_info_{idx+1}"
    # saving_dir = f"{hydra_output_dir}/{llm_folder}/train/classic/{cfg.corpus.retriever_str}/{num_doc}_doc"
    # saving
    saving_dir = HydraConfig().get().runtime.output_dir
    logger.info(f"Output directory: {saving_dir}")
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)

    
    # MPT has a different answer string in the prompt
    answer_string_in_prompt = "### Response:" if 'mpt' in llm_id else "Answer:"

    all_info = []  
    for idx, prompt_batch in enumerate(tqdm(prompt_dataloader)):
        prompts = prompt_batch['prompt']
        # add breakpoint
        # import pdb; pdb.set_trace()
        generated_output = llm.generate_batch(prompts, max_new_tokens=cfg.llm.max_new_tokens,
                                              top_k = cfg.llm.top_k, top_p = cfg.llm.top_p,
                                              presence_penalty=cfg.llm.presence_penalty)

        generated_answers = []
        for output in generated_output:
            response = output.strip()
            generated_answers.append(response)

        prompt_batch['generated_answer'] = generated_answers
        all_info.append(prompt_batch)
        
        if (idx + 1) % save_every == 0 or (idx + 1) == len(prompt_dataloader):
            logger.info(f"Saving at batch {idx + 1}...")
            file_name = f"{saving_dir}/gold_at{gold_pos}{rand_str}{answerless_str}_info_{idx+1}.pkl"
            write_pickle(all_info, file_name)
            logger.info(f"Saved to {file_name}")
            all_info = []


@hydra.main(version_base=None, config_path="../conf", config_name="generation_config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    logger = logging.getLogger(__name__)
    
    validate_config(cfg)
    logger.info(f"validated configuration {cfg}")
    
    # Set seed
    seed_everything(cfg.seed)

    # import pdb; pdb.set_trace()

    logger.info("Loading LLM...")
    llm_id = cfg.llm.llm_id
    # llm = LLM(
    #     llm_id, device, quantization_bits=4, 
    #     model_max_length=cfg.llm.model_max_length
    # )
    if cfg.llm.debug: 
        llm = None
        logger.debug("LLM not loaded")
    else:
        if getattr(cfg.llm, "mode", "offline") == "server":
            llm = VLLMServerClient(
                server_url=cfg.llm.server_url,
                served_model_name=llm_id,
            )
        else:
            llm = VLLMWrapper(
                llm_id, device,
                model_max_length=cfg.llm.model_max_length,
                tensor_parallel_size=cfg.llm.tensor_parallel_size,
                quantization_bits=cfg.llm.quantization_bits,
                gpu_memory_utilization=cfg.llm.gpu_memory_utilization,
                eager_mode=cfg.llm.eager_mode,
                parse_reasoning=cfg.llm.parse_reasoning,
            )
    tokenizer = AutoTokenizer.from_pretrained(
        llm_id, 
        padding_side="left", 
        truncation_side="left", 
        model_max_length=cfg.llm.model_max_length
    )
    logger.info("LLM loaded")

    logger.info("Loading corpus and search results...")
    corpus, full_to_subset_idx_map = load_corpus(cfg)
    search_results = load_search_results(cfg)
    logger.info("Corpus and search results loaded")

    logger.info("Loading prompt dataset...")
    prompt_dataloader = initialize_dataset_and_loader(
        cfg, corpus, full_to_subset_idx_map, search_results, tokenizer
    )
    logger.info("Prompt dataset loaded")

    print_info(cfg)
    generate_and_save(cfg, llm, prompt_dataloader)



if __name__ == "__main__":
    load_dotenv()
    OmegaConf.register_new_resolver("sanitize_path", lambda x: x.split("/")[1] if "/" in x else x)
    main()
