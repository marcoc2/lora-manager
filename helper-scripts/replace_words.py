# Exemplo de plugin: replace_words.py

def get_parameters():
    return [
        {
            "name": "replace_dict",
            "label": "Dicionário de Substituições",
            "type": "dict",
            "default": "{'old_word': 'new_word'}"
        }
    ]

def process_files(input_path, **kwargs):
    import os
    import ast
    # Se replace_dict veio como string, pode ser convertido com ast.literal_eval
    replace_dict = kwargs.get("replace_dict")
    if isinstance(replace_dict, str):
        replace_dict = ast.literal_eval(replace_dict)
    
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            for old, new in replace_dict.items():
                content = content.replace(old, new)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
