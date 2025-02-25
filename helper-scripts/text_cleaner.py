# Plugin para transformações avançadas de texto

def get_parameters():
    return [
        {
            "name": "operations",
            "label": "Lista de operações a realizar",
            "type": "dict",
            "default": """[
                "strip_lines",            # Remove espaços no início/fim das linhas
                "remove_empty_lines",     # Remove linhas vazias
                "normalize_spaces",       # Normaliza espaços entre palavras
                "smart_title"            # Capitalização inteligente de títulos
            ]"""
        },
        {
            "name": "case_transform",
            "label": "Transformação de caso (none/upper/lower/title/smart_title)",
            "type": "str",
            "default": "none"
        },
        {
            "name": "remove_duplicates",
            "label": "Remover linhas duplicadas (true/false)",
            "type": "str",
            "default": "false"
        },
        {
            "name": "line_prefix",
            "label": "Prefixo para cada linha",
            "type": "str",
            "default": ""
        },
        {
            "name": "line_suffix",
            "label": "Sufixo para cada linha",
            "type": "str",
            "default": ""
        }
    ]

def process_files(input_path, **kwargs):
    """
    Aplica transformações avançadas em arquivos de texto.
    """
    import os
    import ast
    import re
    from datetime import datetime

    def smart_title(text):
        """Capitalização inteligente que ignora artigos/preposições menores"""
        # Palavras que não devem ser capitalizadas (exceto no início)
        small_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'for', 'if', 'in', 
                      'of', 'on', 'or', 'the', 'to', 'via', 'vs'}
        
        words = text.lower().split()
        # Primeira palavra sempre capitalizada
        if words:
            words[0] = words[0].capitalize()
        
        # Demais palavras: capitaliza se não for uma "small word"
        for i in range(1, len(words)):
            if words[i] not in small_words:
                words[i] = words[i].capitalize()
                
        return ' '.join(words)

    def normalize_spaces(text):
        """Normaliza espaços no texto"""
        # Remove espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        # Garante espaço após pontuação
        text = re.sub(r'([.,!?;:](?:\s*["\'])?)\s*', r'\1 ', text)
        return text.strip()

    # Processa os parâmetros
    operations = kwargs.get('operations', '[]')
    if isinstance(operations, str):
        operations = ast.literal_eval(operations)
    
    case_transform = kwargs.get('case_transform', 'none')
    remove_duplicates = kwargs.get('remove_duplicates', 'false').lower() == 'true'
    line_prefix = kwargs.get('line_prefix', '')
    line_suffix = kwargs.get('line_suffix', '')

    for filename in os.listdir(input_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_path, filename)
            
            # Lê o arquivo
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Processa linha por linha
            processed_lines = []
            for line in lines:
                if 'strip_lines' in operations:
                    line = line.strip()
                
                if case_transform != 'none':
                    if case_transform == 'upper':
                        line = line.upper()
                    elif case_transform == 'lower':
                        line = line.lower()
                    elif case_transform == 'title':
                        line = line.title()
                    elif case_transform == 'smart_title':
                        line = smart_title(line)

                if 'normalize_spaces' in operations:
                    line = normalize_spaces(line)

                if line_prefix or line_suffix:
                    line = f"{line_prefix}{line}{line_suffix}"

                if line or 'remove_empty_lines' not in operations:
                    processed_lines.append(line)

            # Remove duplicatas se solicitado
            if remove_duplicates:
                processed_lines = list(dict.fromkeys(processed_lines))

            # Salva o resultado
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(processed_lines) + '\n')