# This class provides a convenient interface for generating text from a language model 
# that can be used in a variety of applications, such as chatbots or virtual assistants.
#
# it downloads a language model from the Hugging Face model hub and creates an instance of the Model class 
# provided by the pyllamacpp package to generate text based on the prompt generated from the user's input. 
# It uses the jinja2 package to create prompts from a template 
# and the duckduckgo_search package for language translation.

from huggingface_hub import hf_hub_download
from jinja2 import Environment, FileSystemLoader
from pyllamacpp.model import Model

from pyagents.utils import get_os_name, string_to_array, translateTo


class CommandParser:
    # Initializes an instance of the CommandParser class with a specified model name, repository ID, 
    # and prompt template.
    def __init__(
        self, 
        model_name="ggjt-model.bin", 
        repo_id="LLukas22/gpt4all-lora-quantized-ggjt", 
        prompt_template="parse_command"
    ):
        self.repo_id = repo_id
        self.ggml_model = model_name
        self.llama_context_params = {
            "n_ctx": 3000,
        }
        self.gpt_params = {
            "n_predict": 20,
            "n_threads": 5,
            "temp": 0.2,
        }
        self.prompt_template = prompt_template

        #Download the model (if not already downloaded)
        hf_hub_download(self.repo_id, filename=self.ggml_model)
        self.model = Model(ggml_model=self.ggml_model, **self.llama_context_params)

    # Generates a prompt string for the model to use as input. The prompt is created by rendering a Jinja2 template 
    # using the command, user input, and the name of the current operating system.
    def _generatePrompt(self, command, user_input, prompt_template=None):
        if prompt_template is not None:
            prompt_template = self.prompt_template

        # get the template and compile the prompt
        env = Environment(loader=FileSystemLoader('prompt_templates'))
        template = env.get_template(f"{prompt_template}.j2")

        # Translate the product description to english to facilitate the model generating the fun fact
        rendered_prompt = template.render(
            command=command,
            prompt=user_input,
            os=get_os_name(),
        )
        return rendered_prompt    

    # Generates text from the model using the prompt generated from the user's input.
    def GenerateText(self, command, user_input, prompt_template=None):
        prompt = self._generatePrompt(command=command, user_input=user_input, prompt_template=prompt_template)
        generated_text = self.model.generate(
            prompt,
            new_text_callback=None, # type: ignore
            verbose=False,
            **self.gpt_params,
        )

        # Strip the prompt from the generated text
        lines = generated_text.split('\n')
        last_line = lines[-1]
        result = string_to_array(last_line)
        return result

    # Wrapper function for the translateTo function in utils.py
    def translateTo(self, text, lang="en"):
        return translateTo(text, lang)
 


