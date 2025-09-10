import torch
from transformers import AutoTokenizer
from vllm import LLM as vLLM_Engine, SamplingParams
from typing import List, Optional, Union, Any

class VLLMWrapper:
    """
    A wrapper for vLLM engine that provides the same interface as the original LLM class
    but with significantly improved performance through PagedAttention and batching.
    
    Attributes:
        model_id (str): Identifier for the model to load.
        quantization_bits (Optional[int]): Number of bits for quantization (4 or 8).
        stop_list (Optional[List[str]]): List of tokens where generation should stop.
        model_max_length (int): Maximum length of the model inputs.
    """
    
    def __init__(
        self, 
        model_id: str, 
        device: str = 'cuda', 
        quantization_bits: Optional[int] = None, 
        stop_list: Optional[List[str]] = None, 
        model_max_length: int = 4096,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        eager_mode: bool = False,
        parse_reasoning:bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.model_max_length = model_max_length
        self.tensor_parallel_size = tensor_parallel_size
        
        self.stop_list = stop_list or ['\\nHuman:', '\\n```\\n', '\\nQuestion:', '<|endoftext|>', '\\n']
        
        # Configure quantization
        quantization = None
        if quantization_bits == 4:
            quantization = "bitsandbytes"  # vLLM supports AWQ for 4-bit
        elif quantization_bits == 8:
            raise Exception("fp8 mode not supported")
            quantization = "fp8"  # vLLM supports FP8 for 8-bit
            
        # Initialize vLLM engine
        self.llm = vLLM_Engine(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=torch.bfloat16,
            max_model_len=model_max_length,
            quantization=quantization,
            trust_remote_code=True,
            enforce_eager=eager_mode,
        )
        self.parse_reasoning = parse_reasoning
        if parse_reasoning:
            from vllm.reasoning import Qwen3ReasoningParser
            if model_id.startswith('Qwen/Qwen3'):
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                self.reasoning_parser = Qwen3ReasoningParser(tokenizer)
            else:
                raise NotImplementedError(f"Reasoning parsing not implemented for model {model_id}")

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 15,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> List[str]:
        """
        Generates text based on the given prompt.
        
        Args:
            prompt (str): Input text prompt for generation.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature (0.0 for greedy).
            top_p (float): Top-p sampling parameter.
        
        Returns:
            List[str]: The generated text responses (only new tokens, not including prompt).
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=self.stop_list,
            repetition_penalty=1.1
        )
        
        outputs = self.llm.generate([prompt], sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def generate_batch(
        self, 
        prompts: List[str], 
        max_new_tokens: int = 15,
        temperature: float = 0.0,
        top_p: float = 1.0
    ) -> List[str]:
        """
        Generates text for multiple prompts in batch (much faster than sequential).
        
        Args:
            prompts (List[str]): List of input text prompts for generation.
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature (0.0 for greedy).
            top_p (float): Top-p sampling parameter.
        
        Returns:
            List[str]: The generated text responses (only new tokens, not including prompts).
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop=self.stop_list,
            repetition_penalty=1.1
        )
        
        outputs = self.llm.generate(prompts, sampling_params)
        if self.parse_reasoning:
            # Parse reasoning steps if enabled
            parsed_outputs = []
            for output in outputs:
                text = output.outputs[0].text
                reasoning_trace, answer = self.reasoning_parser.extract_reasoning_content(text, None)
                parsed_outputs.append(answer)
            return parsed_outputs
        else:
            return [output.outputs[0].text for output in outputs]