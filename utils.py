import ast
import platform
from duckduckgo_search import ddg_translate

def string_to_array(input_string):
    try:
        array = ast.literal_eval(input_string)
        if isinstance(array, list):
            return array
        else:
            raise ValueError("Input string is not a list.")
    except (SyntaxError, ValueError) as e:
        print(f"Error: {e}")
        return None    

def get_os_name():
    os_name = platform.system()
    if os_name == "Darwin":
        return "MacOS"
    elif os_name == "Windows":
        return "Windows"
    elif os_name == "Linux":
        return "Linux"
    else:
        return "Unknown"

# Translates text to a specified language using the ddg_translate function from the duckduckgo_search package.
def translateTo(text, lang="en"):
    translation = ddg_translate(text, to=lang)
    if len(translation) > 0:
        # only use the first translation
        translation = translation[0]
        if translation["detected_language"] != lang:
            text = translation["translated"]
    return text       