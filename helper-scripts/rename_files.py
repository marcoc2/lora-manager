# Plugin para renomear arquivos em sequência com prefixo personalizado

def get_parameters():
    return [
        {
            "name": "prefix",
            "label": "Prefixo para os arquivos",
            "type": "str",
            "default": "file"
        },
        {
            "name": "extensions",
            "label": "Lista de extensões (ex: ['.txt', '.jpg'])",
            "type": "dict",  # Usando dict type para lista
            "default": "['.txt', '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']"
        },
        {
            "name": "start_number",
            "label": "Número inicial para a sequência",
            "type": "int",
            "default": "1"
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
    Renomeia arquivos em sequência usando um prefixo e número.
    
    Args:
        input_path (str): Caminho do diretório contendo os arquivos
        **kwargs: Deve conter:
            - prefix: prefixo para os novos nomes
            - extensions: lista de extensões de arquivo a processar
            - start_number: número inicial para a sequência
            - create_backup: se deve criar backup dos arquivos originais
    """
    import os
    import ast
    import shutil
    from datetime import datetime
    
    # Processa os parâmetros
    prefix = kwargs.get('prefix', 'file')
    extensions = kwargs.get('extensions', "['.txt']")
    if isinstance(extensions, str):
        extensions = ast.literal_eval(extensions)
    start_number = int(kwargs.get('start_number', 1))
    create_backup = kwargs.get('create_backup', 'true').lower() == 'true'
    
    # Converte extensões para minúsculas para comparação case-insensitive
    extensions = [ext.lower() for ext in extensions]
    
    # Cria pasta de backup se necessário
    if create_backup:
        backup_folder = os.path.join(input_path, f'backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(backup_folder, exist_ok=True)
    
    # Lista e ordena os arquivos que correspondem às extensões
    files = []
    for filename in os.listdir(input_path):
        file_path = os.path.join(input_path, filename)
        if (os.path.isfile(file_path) and 
            os.path.splitext(filename)[1].lower() in extensions):
            files.append(filename)
    files.sort()  # Ordena os arquivos alfabeticamente
    
    # Processa cada arquivo
    for i, filename in enumerate(files, start=start_number):
        old_path = os.path.join(input_path, filename)
        _, ext = os.path.splitext(filename)
        new_name = f"{prefix}_{i:03d}{ext.lower()}"  # Formato: prefixo_001.ext
        new_path = os.path.join(input_path, new_name)
        
        # Faz backup se necessário
        if create_backup:
            shutil.copy2(old_path, os.path.join(backup_folder, filename))
        
        # Renomeia o arquivo
        os.rename(old_path, new_path)