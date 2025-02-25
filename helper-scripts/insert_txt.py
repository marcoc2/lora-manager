# Plugin para inserir texto no início dos arquivos

def get_parameters():
    return [
        {
            "name": "prefix_text",
            "label": "Texto a ser adicionado no início",
            "type": "str",
            "default": ""
        }
    ]

def process_files(input_path, **kwargs):
    """
    Adiciona um texto específico no início de cada arquivo.
    
    Args:
        input_path (str): Caminho do diretório contendo os arquivos a serem processados
        **kwargs: Deve conter 'prefix_text'
    """
    import os
    
    prefix_text = kwargs.get('prefix_text', '')
    
    if not prefix_text:
        raise ValueError("O parâmetro 'prefix_text' é obrigatório")
    
    for filename in os.listdir(input_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_path, filename)
            
            # Lê o conteúdo do arquivo
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Adiciona o texto no início
            new_content = prefix_text + content
            
            # Escreve o novo conteúdo no arquivo
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)