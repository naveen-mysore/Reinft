import json
import re
from typing import Optional, Tuple


class PromptManager:
    """
    Class to manage prompt templates and generation,
    using a conversation-like format suitable for LLaMA-based models.
    """

    def __init__(self):
        ########################################################################
        # 1) Store your system instructions in a single string
        ########################################################################
        self.system_instructions = (
            "You are a helpful nutrition assistant that calculates the total carbohydrates in a meal.\n"
            "For the given query including a meal description, think step by step as follows:\n"
            "1. Parse the meal description into discrete food or beverage items along with their serving size. "
            "   If the serving size of any item in the meal is not specified, assume it is a single standard serving "
            "   based on common nutritional guidelines (e.g., USDA). Ignore additional information that doesn't relate "
            "   to the item name and serving size.\n"
            "2. For each food or beverage item in the meal, calculate the amount of carbohydrates in grams for the "
            "   specific serving size.\n"
            '3. For the total carbohydrates, respond with just the numeric amount of carbohydrates without extra text. '
            '   If you don\'t know the answer, set the value of "total_carbohydrates" to -1.\n'
            "4. Always respond in the user's original natural language of the Query.\n"
            "5. Respond with a dictionary object containing the total carbohydrates in grams as follows:\n"
            '{"total_carbohydrates": total grams of carbohydrates for the serving}\n\n'

            # Example 1
            'Example 1:\n'
            'Query: "This morning, I had a cup of oatmeal with half a sliced banana and a glass of orange juice."\n'
            "Answer:<cot>\n"
            "The meal consists of 1 cup of oatmeal, 1/2 a banana and 1 glass of orange juice.\n"
            "1 cup of oatmeal has 27g carbs.\n"
            "1 banana has 27g carbs so half a banana has (27*(1/2)) = 13.5g carbs.\n"
            "1 glass of orange juice has 26g carbs.\n"
            "So the total grams of carbs in the meal = (27 + 13.5 + 26) = <66.5>g carbs\n"
            "</cot>\n"
            '<answer>Output: {"total_carbohydrates": 66.5}</answer>\n\n'

            # Example 2
            'Example 2:\n'
            'Query: "朝食に、卵2個で作ったスクランブルエッグとトーストを食べました。"\n'
            "Answer:<cot>\n"
            "その食事は卵2個で作ったスクランブルエッグと1枚のトーストで構成されています。\n"
            "卵2個で作ったスクランブルエッグには2gの炭水化物があります。\n"
            "1枚のトーストには13gの炭水化合物があります。\n"
            "よって、その食事に含まれる炭水化合物の総量は = (2 + 13) = <15>g です\n"
            "</cot>\n"
            '<answer>Output: {"total_carbohydrates": 15}</answer>\n\n'

            # Example 3
            'Example 3:\n'
            'Query: "半个花生酱和果酱三明治。"\n'
            "Answer:<cot>\n"
            "这份餐点由1/2的花生酱和果酱三明治组成。\n"
            "1个花生酱和果酱三明治含有50.6g碳水化合物，所以半个花生酱和果酱三明治\n"
            "含有(50.6*(1/2)) = <25.3>g碳水化合物\n"
            "因此，这份餐所含的碳水化合物总量 = <25.3>g碳水化合物\n"
            "</cot>\n"
            '<answer>Output: {"total_carbohydrates": 25.3}</answer>'
        )

        ########################################################################
        # 2) Minimal base prompt (training mode) that does NOT include system text
        ########################################################################
        self.training_base_prompt = "<start_of_turn>user\n"

    def build_prompt(self, query: str, mode: str = "training") -> str:
        """
        Build the complete prompt, either for training or inference.

        Args:
            query: The user's meal query text
            mode: "training" or "inference"

        Returns:
            Full prompt text
        """
        if mode == "inference":
            # For inference, full system role with instructions + examples
            # User role contains only the actual query
            return (
                "<start_of_turn>system\n"
                f"{self.system_instructions}\n"
                "<end_of_turn>\n"
                "<start_of_turn>user\n"
                f'Query: "{query}"\n'
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
                "Answer:<cot>\n"
            )
        else:
            # Training mode: minimal prompt (no system instructions)
            return (
                    self.training_base_prompt
                    + f'Query: "{query}"\n'
                      "<end_of_turn>\n"
                      "<start_of_turn>model\n"
                      "Answer:<cot>\n"
            )

    def build_training_sample(
            self,
            query: str,
            cot: str,
            answer_value: str,
            eos_token: str
    ) -> str:
        """
        Build a complete training sample with prompt, COT, and final answer.

        Args:
            query: The user's meal query text
            cot: The chain-of-thought reasoning text
            answer_value: The carbohydrate value to include in the final answer
            eos_token: The model's end-of-sequence token

        Returns:
            Complete text for one training example.
        """
        prompt = self.build_prompt(query, mode="training")
        result_json = {"total_carbohydrates": answer_value.strip()}

        return (
                prompt
                + (cot if cot else "")
                + "\n</cot>\n"
                + "<answer>Output: "
                + json.dumps(result_json)
                + "</answer><end_of_turn>"
                + eos_token
        )

    def build_inference_prompt(self, query: str) -> str:
        """
        Convenience method for building an inference-time prompt
        with system instructions included.

        Args:
            query: The user's meal query text

        Returns:
            The full prompt (including system text) to feed the LLaMA model at inference
        """
        return self.build_prompt(query, mode="inference")

    def extract_model_block(self, generated_text: str) -> str:
        """
        Extract the <start_of_turn>model...</end_of_turn> block from the generated text.

        Args:
            generated_text: The full text generated by the model

        Returns:
            The extracted model block, or an empty string if not found.
        """
        start_marker = "<start_of_turn>model"
        end_marker = "<end_of_turn>"

        start_idx = generated_text.find(start_marker)
        if start_idx == -1:
            return ""

        end_idx = generated_text.find(end_marker, start_idx)
        if end_idx == -1:
            return generated_text[start_idx:]
        else:
            return generated_text[start_idx:end_idx + len(end_marker)]

    def parse_cot_and_answer(self, text: str, verbose=False, log_fn=None) -> Tuple[str, str]:
        """
        Parse a generated response to extract chain-of-thought (CoT) and final answer.
        Handles multiple formats that might appear in model outputs.

        Args:
            text: The full text response
            verbose: Whether to print debug information
            log_fn: Function to use for logging (e.g., accelerator.print)

        Returns:
            Tuple of (chain_of_thought, answer)
        """
        # Clean up the text - remove special tokens that might interfere
        cleaned_text = re.sub(r'<pad>|<bos>|<eos>', '', text).strip()

        if verbose and log_fn:
            log_fn(f"Parsing text (length: {len(cleaned_text)})")
            # Show a brief preview if text is long
            if len(cleaned_text) > 100:
                log_fn(f"Text preview: {cleaned_text[:50]}...{cleaned_text[-50:]}")

        # Initialize empty results
        chain_of_thought = ""
        answer = ""

        # METHOD 1: Extract from XML-style tags
        cot_match = re.search(r'<cot>(.*?)</cot>', cleaned_text, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned_text, re.DOTALL)

        if cot_match:
            chain_of_thought = cot_match.group(1).strip()
            if verbose and log_fn:
                log_fn("Found CoT using tag pattern")

        if answer_match:
            answer = answer_match.group(1).strip()
            if verbose and log_fn:
                log_fn("Found answer using tag pattern")

        # METHOD 2: Look for "final_answer:" sections
        if not answer:
            final_ans_pattern = re.search(
                r'final_answer:\s*(.*?)(?:\n\s*chain_of_thought:|$)',
                cleaned_text, re.DOTALL | re.IGNORECASE
            )
            if final_ans_pattern:
                answer = final_ans_pattern.group(1).strip()
                if verbose and log_fn:
                    log_fn("Found answer using 'final_answer:' pattern")

        # METHOD 3: Look for "Output:" sections with JSON
        if not answer:
            output_json_pattern = re.search(
                r'Output:\s*({.*?"total_carbohydrates".*?})',
                cleaned_text, re.DOTALL
            )
            if output_json_pattern:
                answer = "Output: " + output_json_pattern.group(1).strip()
                if verbose and log_fn:
                    log_fn("Found answer using 'Output:' with JSON pattern")

        # METHOD 4: If still no chain of thought, try to find a "step by step" pattern
        if not chain_of_thought:
            step_patterns = [
                r"Let\'s think step by step\.(.*?)(?:final_answer:|Output:|$)",
                r"Let\'sthinkstepbystep\.(.*?)(?:final_answer:|Output:|$)",
                r"Thinking step by step:(.*?)(?:final_answer:|Output:|$)",
                r"Step by step analysis:(.*?)(?:final_answer:|Output:|$)",
            ]
            for pattern in step_patterns:
                match = re.search(pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
                if match:
                    chain_of_thought = match.group(1).strip()
                    if verbose and log_fn:
                        log_fn(f"Found CoT using pattern: {pattern[:20]}...")
                    break

        # METHOD 5: Last resort - look for any carb value patterns in the text
        if not answer:
            # Look for total_carbohydrates in various formats
            carb_patterns = [
                r'"total_carbohydrates"\s*:\s*"?(\d+\.?\d*)"?',  # JSON format
                r'total\s+carbs.*?=\s*<?(\d+\.?\d*)>?\s*g',  # equals format with optional brackets
                r'<(\d+\.?\d*)>\s*grams',  # bracketed value with "grams"
                r'Output:\s*\(?(\d+\.?\d*)\)?'  # "Output:" with a number
            ]

            for pattern in carb_patterns:
                match = re.search(pattern, cleaned_text, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1)
                    answer = f'Output: {{"total_carbohydrates": "{value}"}}'
                    if verbose and log_fn:
                        log_fn(f"Found answer using carb value pattern: {pattern[:20]}...")
                    break

        if verbose and log_fn:
            log_fn("Final extracted values:")
            log_fn(f"CoT length: {len(chain_of_thought)}")
            log_fn(f"Answer: {answer}")

        return chain_of_thought, answer

    def extract_carbs_from_answer(self, answer_text: str) -> Optional[float]:
        """
        Extract the carbohydrate value from the answer text, handling multiple formats.

        Args:
            answer_text: The text of the answer section

        Returns:
            The extracted carbohydrate value as a float, or None if not found
        """
        if not answer_text or not answer_text.strip():
            return None

        # Try direct pattern matching first
        pattern = r'"total_carbohydrates"\s*:\s*"?(-?\d+(?:\.\d+)?)"?'
        match = re.search(pattern, answer_text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        # Look for numbers inside JSON format
        json_pattern = r'{.*?}'
        json_match = re.search(json_pattern, answer_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                json_str = re.sub(r'\s+', ' ', json_str)
                data = json.loads(json_str)
                if "total_carbohydrates" in data:
                    carb_value = data["total_carbohydrates"]
                    if isinstance(carb_value, (int, float)):
                        return float(carb_value)
                    elif isinstance(carb_value, str):
                        return float(carb_value)
            except (json.JSONDecodeError, ValueError):
                pass

        # Look for numbers inside <> brackets
        angle_pattern = r'<(\d+(?:\.\d+)?)>'
        angle_match = re.search(angle_pattern, answer_text)
        if angle_match:
            try:
                return float(angle_match.group(1))
            except ValueError:
                pass

        # Try "Output:" or "="
        for marker in ["Output:", "="]:
            if marker in answer_text:
                after_marker = answer_text.split(marker)[-1]
                num_pattern = r'(\d+(?:\.\d+)?)'
                num_match = re.search(num_pattern, after_marker)
                if num_match:
                    try:
                        return float(num_match.group(1))
                    except ValueError:
                        pass

        # Last resort: any number in the text
        num_pattern = r'(\d+(?:\.\d+)?)'
        num_match = re.search(num_pattern, answer_text)
        if num_match:
            try:
                return float(num_match.group(1))
            except ValueError:
                pass

        return None