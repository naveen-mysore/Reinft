import json
import re
from typing import Optional, Tuple


class PromptManager:
    """
    Class to manage prompt templates and generation,
    using a conversation-like format compatible with LLaMA-based models.
    
    We adopt a standard Llama 2 style prompt:
      <s>[INST] ...system instructions... ...user prompt... [/INST]
      (assistant response) 
      </s>
      
    We still keep <|begin_cot|> / <|end_cot|> around if you want to store
    chain-of-thought for training. 
    """

    def __init__(self):
        # System instructions: what the model should do globally
        self.system_instructions = (
            "You are a nutrition assistant that calculates the total carbohydrates in any given food. When responding, break each item into its components, look up or estimate the carbs per serving for each component, and sum those values to find the total. Clearly explain how you arrived at that total by describing your step-by-step reasoning. Finally, present the total carbohydrate content inside angle brackets (e.g., <10> grams) and respond in the same language as the user's query.\n"
        )

    def build_prompt(self, query, mode="inference"):
        """
        Backward compatibility method that redirects to the appropriate prompt builder.
        
        Args:
            query: The user query
            mode: Either "inference" or "training"
            
        Returns:
            The formatted prompt in Llama style
        """
        if mode == "inference":
            return self.build_inference_prompt(query)
        else:
            # For training mode, just build the inference prompt (without answer)
            # The actual training samples are built with build_training_sample
            return self.build_inference_prompt(query)

    def build_inference_prompt(self, user_query: str) -> str:
        """
        Build the inference-time prompt for Llama style:

        <s>[INST]
          (system instructions + examples)
          User: ...
        [/INST]
        Assistant:
        
        We do not necessarily need <|begin_cot|> at inference time, 
        but if you want the chain of thought, you can still add it.
        """
        return (
            f"<s>[INST] {self.system_instructions}\n"
            f"User: {user_query}\n"
            "[/INST]\n"
            "Assistant:"
        )

    def build_training_sample(
        self,
        query: str,
        cot: str,
        answer_value: str,
        eos_token: str
    ) -> str:
        """
        For supervised fine-tuning, we typically put the system + user
        instructions in a single [INST] block and then let the 'Assistant:'
        block hold the chain-of-thought or final answer. 
        We explicitly add an EOS token at the very end.
        """
        # Llama style system + user block
        prompt_prefix = (
            f"<s>[INST] {self.system_instructions}\n"
            f"User: {query}\n"
            "[/INST]\n"
            "Assistant:<|begin_cot|>\n"
        )

        # If chain-of-thought is empty, we still produce a minimal final answer
        # Ensure we have at least <answer_value> in brackets somewhere
        final_cot = cot.strip() if cot else f"<{answer_value.strip()}>"
        if "<" not in final_cot or ">" not in final_cot:
            final_cot += f" = <{answer_value.strip()}>"

        # Close chain-of-thought, then close the assistant block with </s> or EOS
        prompt_suffix = "<|end_cot|>\n</s>"

        return prompt_prefix + final_cot + prompt_suffix + eos_token

    def extract_model_block(self, generated_text: str) -> str:
        """
        Extract the model's portion from a fully generated text that may contain
        the system + user instructions. Looks for the first 'Assistant:' after the
        last [INST].
        """
        # We can look for the last [INST], then find "Assistant:" 
        # and return everything from there onward.
        pattern = r"\[INST\].*?[/INST]\s*Assistant:(.*)"
        match = re.search(pattern, generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return generated_text  # fallback

    def parse_cot_and_answer(self, text: str, verbose=False, log_fn=None) -> Tuple[str, str]:
        """
        Parse chain-of-thought (CoT) and final <X> answer from the text.
        1) Extract chain-of-thought from <|begin_cot|> ... <|end_cot|>.
           If none, fallback to entire assistant block.
        2) Check the *last two tokens* for a bracketed numeric answer, e.g. <123.45>.
        3) Return (chain_of_thought, answer_str).
           If no bracket is found, answer_str is "".
        """
        # Extract the model's response part first - ensure we're only looking at the assistant's response
        text_to_parse = self.extract_model_block(text)
        
        # Debug logging to see what we're parsing
        if verbose and log_fn:
            log_fn(f"Parsing text (truncated): {text_to_parse[:100]}...")
        
        # Extract CoT content
        cot_pattern = r"<\|begin_cot\|>(.*?)<\|end_cot\|>"
        cot_match = re.search(cot_pattern, text_to_parse, re.DOTALL)
        if cot_match:
            chain_of_thought = cot_match.group(1).strip()
            if verbose and log_fn:
                log_fn("Found CoT between <|begin_cot|> and <|end_cot|>.")
        else:
            chain_of_thought = text_to_parse
            if verbose and log_fn:
                log_fn("No <|begin_cot|> found; using entire text as CoT fallback.")
        
        # Split the CoT by whitespace and look in the last 1-2 tokens for bracket
        tokens = chain_of_thought.split()
        if len(tokens) == 0:
            # no content => no answer
            return (chain_of_thought, "")

        # gather last 2 tokens (or 1 if there's only 1)
        last_tokens = tokens[-2:] if len(tokens) >= 2 else tokens[-1:]
        snippet = " ".join(last_tokens)
        
        match = re.search(r"<([\-\d\.]+)>", snippet)
        if match:
            answer_str = match.group(1).strip()
            if verbose and log_fn:
                log_fn(f"Found bracketed numeric answer in the last tokens: {answer_str}")
        else:
            answer_str = ""
            if verbose and log_fn:
                log_fn("No bracket found in last tokens => empty answer")
        
        return chain_of_thought, answer_str

    def extract_carbs_from_answer(self, answer_text: str) -> Optional[float]:
        """
        Attempt to parse the bracketed float, e.g. <123.45>.
        If not found, fallback to numeric parse from any substring in 'answer_text'.
        Returns None if no valid parse.
        """
        if not answer_text:
            return None

        # 1) Direct bracket parse
        bracket_vals = re.findall(r"<([\-\d\.]+)>", answer_text)
        if bracket_vals:
            # take the last bracket
            val_str = bracket_vals[-1]
            try:
                return float(val_str)
            except ValueError:
                pass

        # 2) fallback: parse numeric from entire string
        try:
            clean = re.sub(r"[^\d\.]+", "", answer_text)
            if clean:
                return float(clean)
        except ValueError:
            pass

        return None