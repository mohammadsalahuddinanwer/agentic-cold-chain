"""Llama 3.1 8B Instruct wrapper with Hugging Face Transformers."""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Llama31LLM:
    """
    Wrapper for Meta's Llama 3.1 8B Instruct model.
    Supports 8-bit quantization for reduced memory usage.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        load_in_8bit: bool = True,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        use_pipeline: bool = True,
    ):
        """
        Initialise the Llama 3.1 model.

        Args:
            model_id: Hugging Face model identifier.
            load_in_8bit: Whether to use 8-bit quantization (reduces VRAM to ~6GB).
            device_map: Device mapping strategy ('auto', 'cuda', 'cpu').
            torch_dtype: Data type for model weights.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0.0 = greedy).
            top_p: Nucleus sampling threshold.
            use_pipeline: Use `pipeline` API (simpler) vs `AutoModel` (more control).
        """
        self.model_id = model_id
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_pipeline = use_pipeline

        logger.info(f"Loading Llama 3.1 model: {model_id}")
        logger.info(f"8-bit quantization: {load_in_8bit}")

        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer with optional quantization."""
        bnb_config = None
        if self.load_in_8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            logger.info("8-bit quantization enabled")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_pipeline:
            # Use pipeline abstraction – device_map and dtype go inside model_kwargs
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={
                    "torch_dtype": self.torch_dtype,
                    "quantization_config": bnb_config,
                    "device_map": self.device_map,
                },
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            self.model = None
            logger.info("Pipeline loaded successfully")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                quantization_config=bnb_config,
                device_map=self.device_map,
                trust_remote_code=True,
            )
            self.model.eval()
            logger.info("AutoModel loaded successfully")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: User prompt or instruction.
            system_prompt: Optional system instruction (e.g., role setting).
            max_new_tokens: Override default max tokens.
            temperature: Override default temperature.
            top_p: Override default top_p.
            stop_strings: List of strings to stop generation.

        Returns:
            Generated text response.
        """
        # Build messages in Llama 3.1 chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if self.use_pipeline:
            # Pipeline generation
            outputs = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                top_p=top_p if top_p is not None else self.top_p,
                do_sample=(temperature or self.temperature) > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # FIX 2: this branch accepted stop_strings in the signature and
                # then silently dropped it, so a stop sequence requested by the
                # caller had no effect at all. transformers requires the tokenizer
                # to be passed alongside stop_strings.
                **({"stop_strings": stop_strings, "tokenizer": self.tokenizer}
                   if stop_strings else {}),
            )

            # Extract the generated text
            generated_text = outputs[0]["generated_text"]

            # Remove the original prompt from the response
            assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
            if assistant_header in generated_text:
                response = generated_text.split(assistant_header)[-1].strip()
                response = response.replace("<|eot_id|>", "").strip()
                return response

            return generated_text

        else:
            # AutoModel generation
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens or self.max_new_tokens,
                    temperature=temperature if temperature is not None else self.temperature,
                    top_p=top_p if top_p is not None else self.top_p,
                    do_sample=(temperature or self.temperature) > 0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    # FIX 2 (same branch): transformers raises unless the tokenizer
                    # accompanies stop_strings, and passing stop_strings=None is
                    # not the same as omitting it on some versions.
                    **({"stop_strings": stop_strings, "tokenizer": self.tokenizer}
                       if stop_strings else {}),
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            return response.strip()

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response with tool/function calling support.

        Args:
            prompt: User query.
            tools: List of tool definitions (name, description, parameters).
            system_prompt: Optional system instruction.
            max_new_tokens: Override default.

        Returns:
            Dict with either 'response' (text) or 'tool_call' (function call).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template with tools
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device if self.model else "cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        )

        # Parse tool call if present
        import json
        try:
            # Look for JSON tool call in response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                tool_json = response[start:end]
                tool_call = json.loads(tool_json)
                if "name" in tool_call and "arguments" in tool_call:
                    return {"tool_call": tool_call}
        except json.JSONDecodeError:
            pass

        return {"response": response}

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in batch."""
        return [self.generate(p, **kwargs) for p in prompts]


# Singleton instance for reuse
_llm_instance = None


def get_llm(**kwargs) -> Llama31LLM:
    """Get or create a singleton Llama 3.1 instance."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = Llama31LLM(**kwargs)
    return _llm_instance