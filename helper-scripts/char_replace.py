# Plugin para substituição genérica de caracteres repetidos

def get_parameters():
    return [
        {
            "name": "replace_dict",
            "label": "Dicionário de substituições {'char': 'replacement'}",
            "type": "dict",
            "default": "{'*': '', '#': ''}"  # Exemplo: remove asteriscos e hashtags
        },
        {
            "name": "min_repetitions",
            "label": "Número mínimo de repetições para substituir",
            "type": "int",
            "default": "2"  # Por padrão, substitui quando encontrar 2 ou mais caracteres
        },
        {
            "name": "create_backup",
            "label": "Criar pasta de backup (true/false)",
            "type": "str",
            "default": "true"
        }
    ]

def process_files(input_path, **kwargs):
    """
    Substitui sequências de caracteres repetidos nos arquivos.
    
    Args:
        input_path (str): Caminho do diretório contendo os arquivos a serem processados
        **kwargs: Deve conter:
            - replace_dict: dicionário com {'caractere': 'substituição'}
            - min_repetitions: número mínimo de repetições para fazer a substituição
            - create_backup: se deve criar backup dos arquivos originais
    """
    import os
    import ast
    import shutil
    from datetime import datetime
    
    # Processa os parâmetros
    replace_dict = kwargs.get('replace_dict', {'*': '', '#': ''})
    if isinstance(replace_dict, str):
        replace_dict = ast.literal_eval(replace_dict)
        
    min_repetitions = int(kwargs.get('min_repetitions', 2))
    create_backup = kwargs.get('create_backup', 'true').lower() == 'true'
    
    # Cria pasta de backup se necessário
    if create_backup:
        backup_folder = os.path.join(input_path, f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(backup_folder, exist_ok=True)
    
    # Processa cada arquivo
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_path, filename)
            
            # Faz backup se necessário
            if create_backup:
                shutil.copy2(file_path, os.path.join(backup_folder, filename))
            
            # Lê o conteúdo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Aplica as substituições
            cleaned_content = content
            for char, replacement in replace_dict.items():
                # Continua substituindo enquanto houver sequências do tamanho mínimo
                pattern = char * min_repetitions
                while pattern in cleaned_content:
                    cleaned_content = cleaned_content.replace(pattern, replacement)
            
            # Salva o resultado
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)